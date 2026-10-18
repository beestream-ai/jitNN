package org.beehive.jitnn.ops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.engine.Storage;
import org.beehive.jitnn.kernels.FusedKernels;
import org.beehive.jitnn.kernels.FoldKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Rewrites a recorded trace before it is compiled, fusing a producer into its only consumer:
 * <ul>
 * <li>convolution, then BatchNorm with running statistics (and its ReLU): one convolution on
 * folded weights, with bias and ReLU in its epilogue;</li>
 * <li>linear, then exact GELU: GELU in the GEMM epilogue;</li>
 * <li>add, then ReLU: one element-wise kernel.</li>
 * </ul>
 * Only intermediates nobody else can read are fused away: internal storages (created inside
 * {@code Grad.inference} or by autograd), consumed once, and not requested at this sync. The
 * folded weights are cached as persistent tensors and recomputed, inside the trace that needs
 * them, only when a source parameter or statistic has changed since (storage versions), so a
 * replayed inference step runs no folding kernel. {@code -Djitnn.fuse=false} turns it off.
 */
public final class Fusion {

    private static final boolean ENABLED = !"false".equals(System.getProperty("jitnn.fuse"));

    static final FoldBatchNorm FOLD_BN = new FoldBatchNorm();

    /** Folded weights by the identity of their sources. */
    private static final Map<List<Long>, Folded> FOLDED = new HashMap<>();

    private Fusion() {
    }

    private record Folded(Tensor w, Tensor b, Storage[] sources, long[] versions) {

        boolean current() {
            for (int i = 0; i < sources.length; i++) {
                if (sources[i].version() != versions[i]) {
                    return false;
                }
            }
            return true;
        }
    }

    public static List<Node> rewrite(List<Node> nodes, Tensor[] wanted) {
        if (!ENABLED) {
            return nodes;
        }
        Set<Storage> wantedSet = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor t : wanted) {
            wantedSet.add(t.storage());
        }
        IdentityHashMap<Storage, Integer> uses = new IdentityHashMap<>();
        IdentityHashMap<Storage, Integer> producer = new IdentityHashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (Tensor t : nodes.get(i).inputs) {
                uses.merge(t.storage(), 1, Integer::sum);
            }
            for (Tensor t : nodes.get(i).outputs) {
                producer.put(t.storage(), i);
            }
        }
        // Fused nodes take the producer's place (their inputs are the producer's, plus constants);
        // folding nodes read only parameters this trace does not write, so they run first.
        List<Node> out = new ArrayList<>(nodes);
        List<Node> folds = new ArrayList<>();
        boolean changed = false;
        for (int i = 0; i < out.size(); i++) {
            Node c = out.get(i);
            if (c == null) {
                continue;
            }
            for (Tensor in : c.inputs) {
                Integer p = producer.get(in.storage());
                if (p == null || p >= i || out.get(p) == null || !fusable(in, uses, wantedSet)) {
                    continue;
                }
                Node fused = fuse(out.get(p), c, producer, folds);
                if (fused != null) {
                    out.set(p, fused);
                    out.set(i, null);
                    in.storage().markFused();
                    for (Tensor t : fused.outputs) {
                        producer.put(t.storage(), p);
                    }
                    changed = true;
                    break;
                }
            }
        }
        List<Node> result = new ArrayList<>(folds);
        for (Node n : out) {
            if (n != null) {
                result.add(n);
            }
        }
        // a projection's bias added by the gather of the attention that reads it (MLX on Metal)
        List<Node> biased = FlashAttention.foldQkvBias(result, wantedSet);
        if (biased != result) {
            result = biased;
            changed = true;
        }
        // a linear layer's bias added by the norm that reads it through a residual add (MLX on Metal)
        List<Node> normed = RowNorm.foldResidualBias(result, wantedSet);
        if (normed != result) {
            result = normed;
            changed = true;
        }
        // channels-last storage around library convolutions (MLX on Metal), after BatchNorm folding
        List<Node> layout = ChannelsLast.rewrite(result, wantedSet);
        if (layout != result) {
            result = layout;
            changed = true;
        }
        if (ELEMENTWISE) {
            List<Node> chained = fuseElementwise(result, wantedSet);
            if (chained != null) {
                return chained;
            }
        }
        return changed ? result : nodes;
    }

    /** Element-wise chains into one kernel unless {@code -Djitnn.fuse.elementwise=false}. */
    private static final boolean ELEMENTWISE = !"false".equals(System.getProperty("jitnn.fuse.elementwise"));

    private static boolean elementwise(Node n) {
        if (n.op != Unary.OP && n.op != Binary.OP) {
            return false;
        }
        for (Tensor t : n.inputs) {
            if (t.dtype() != DType.F32) {
                return false;
            }
        }
        return n.out(0).dtype() == DType.F32 && n.out(0).rank() <= 6;
    }

    /** A chain being built: its external inputs, its instructions, and the order of the nodes it absorbed. */
    private static final class Chain {
        final List<Tensor> inputs = new ArrayList<>();
        /** Per instruction: kind, op code, operand a, operand b, as values (input index, or -1 - instruction index). */
        final List<int[]> instructions = new ArrayList<>();
        final List<float[]> params = new ArrayList<>();
        /** The producer chains this one absorbed. */
        final List<Chain> absorbed = new ArrayList<>();
        /** Intermediates the chain also stores (other operators read them): instruction index and tensor. */
        final List<Integer> storeInstr = new ArrayList<>();
        final List<Tensor> stores = new ArrayList<>();
        Tensor output;
    }

    /**
     * Merges unary and binary element-wise operators into {@link ElementwiseFused} chains: a
     * producer joins its consumer when its output has the consumer's shape and only that
     * consumer reads it (internal, not requested), and the chain keeps at most
     * {@link FusedKernels#MAX_INPUTS} inputs and {@link FusedKernels#MAX_INSTRUCTIONS}
     * instructions. A chain runs where its last operator stood, so producers that read persistent
     * storage some operator of the trace writes are not delayed. Returns null when nothing changed.
     */
    private static List<Node> fuseElementwise(List<Node> nodes, Set<Storage> wanted) {
        IdentityHashMap<Storage, Integer> uses = new IdentityHashMap<>();
        Set<Storage> writtenPersistent = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Node n : nodes) {
            for (Tensor t : n.inputs) {
                uses.merge(t.storage(), 1, Integer::sum);
            }
            for (Tensor t : n.outputs) {
                if (t.storage().isPersistent()) {
                    writtenPersistent.add(t.storage());
                }
            }
        }
        IdentityHashMap<Storage, List<Integer>> readers = new IdentityHashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (Tensor t : nodes.get(i).inputs) {
                readers.computeIfAbsent(t.storage(), k -> new ArrayList<>()).add(i);
            }
        }
        IdentityHashMap<Storage, Chain> chainOf = new IdentityHashMap<>();
        IdentityHashMap<Chain, Integer> slot = new IdentityHashMap<>();
        List<Object> order = new ArrayList<>();
        boolean any = false;
        for (int pos = 0; pos < nodes.size(); pos++) {
            Node n = nodes.get(pos);
            if (!elementwise(n)) {
                order.add(n);
                continue;
            }
            Chain c = build(n, pos, chainOf, uses, readers, wanted, writtenPersistent, true);
            if (c.inputs.size() > FusedKernels.MAX_INPUTS) {
                // absorbing producers left too many inputs: this operator starts its own chain
                c = build(n, pos, chainOf, uses, readers, wanted, writtenPersistent, false);
            }
            for (Chain prev : c.absorbed) {
                order.set(slot.get(prev), null);
                chainOf.remove(prev.output.storage());
                if (!c.stores.contains(prev.output)) {
                    prev.output.storage().markFused();
                }
                any = true;
            }
            c.output = n.out(0);
            chainOf.put(n.out(0).storage(), c);
            slot.put(c, order.size());
            order.add(c);
        }
        if (!any) {
            return null;
        }
        List<Node> out = new ArrayList<>();
        IdentityHashMap<Chain, Node> single = new IdentityHashMap<>();
        for (Node n : nodes) {
            if (elementwise(n)) {
                single.put(chainOf.get(n.out(0).storage()), n);
            }
        }
        for (Object o : order) {
            if (o instanceof Node n) {
                out.add(n);
            } else if (o instanceof Chain c) {
                out.add(c.instructions.size() == 1 ? single.get(c) : chainNode(c));
            }
        }
        return out;
    }

    /** The chain of element-wise node n, absorbing its producers' chains when {@code absorb} and they qualify. */
    /** Whether every reader of s other than the node at {@code pos} comes after it. */
    private static boolean readLater(Storage s, int pos, IdentityHashMap<Storage, List<Integer>> readers) {
        for (int r : readers.getOrDefault(s, List.of())) {
            if (r < pos) {
                return false;
            }
        }
        return true;
    }

    private static Chain build(Node n, int pos, IdentityHashMap<Storage, Chain> chainOf, IdentityHashMap<Storage, Integer> uses,
            IdentityHashMap<Storage, List<Integer>> readers, Set<Storage> wanted, Set<Storage> writtenPersistent, boolean absorb) {
        Chain c = new Chain();
        int[] operands = new int[n.inputs.length];
        for (int k = 0; k < n.inputs.length; k++) {
            Tensor in = n.inputs[k];
            Chain prev = absorb ? chainOf.get(in.storage()) : null;
            // dropped: nobody else reads it; stored: others read it, all after this operator, where the chain runs
            boolean drop = prev != null && fusable(in, uses, wanted);
            boolean keep = prev != null && !drop && !in.storage().isPersistent() && readLater(in.storage(), pos, readers);
            if (prev != null) {
                for (Tensor st : prev.stores) {
                    keep &= readLater(st.storage(), pos, readers);
                    drop &= readLater(st.storage(), pos, readers);
                }
            }
            int outputs = c.stores.size() + (prev == null ? 0 : prev.stores.size()) + (keep ? 1 : 0) + 1;
            // the producer's own output shape must be the consumer's (not a reshaped alias of it)
            if (prev != null && (drop || keep) && outputs <= FusedKernels.MAX_OUTPUTS && prev.output.shape().equals(n.out(0).shape()) && in.shape().equals(n.out(0).shape())
                    && !readsWritten(prev, writtenPersistent) && merged(c, prev) <= FusedKernels.MAX_INPUTS
                    && c.instructions.size() + prev.instructions.size() + 1 <= FusedKernels.MAX_INSTRUCTIONS) {
                    // absorb the producer's chain: its inputs and instructions, re-indexed
                    int[] remapIn = new int[prev.inputs.size()];
                    for (int q = 0; q < prev.inputs.size(); q++) {
                        remapIn[q] = inputIndex(c, prev.inputs.get(q));
                    }
                    int shift = c.instructions.size();
                    for (int q = 0; q < prev.instructions.size(); q++) {
                        int[] ins = prev.instructions.get(q).clone();
                        ins[2] = remap(ins[2], remapIn, shift);
                        ins[3] = remap(ins[3], remapIn, shift);
                        c.instructions.add(ins);
                        c.params.add(prev.params.get(q));
                    }
                    for (int q = 0; q < prev.stores.size(); q++) {
                        c.storeInstr.add(prev.storeInstr.get(q) + shift);
                        c.stores.add(prev.stores.get(q));
                    }
                    if (keep) {
                        c.storeInstr.add(c.instructions.size() - 1);
                        c.stores.add(prev.output);
                    }
                operands[k] = -1 - (c.instructions.size() - 1);
                c.absorbed.add(prev);
            } else {
                operands[k] = inputIndex(c, in);
            }
        }
        boolean unary = n.op == Unary.OP;
        c.instructions.add(new int[] { unary ? FusedKernels.UNARY : FusedKernels.BINARY, n.iattr[0], operands[0], unary ? operands[0] : operands[1] });
        c.params.add(new float[] { n.fattr[0], n.fattr[1] });
        c.output = n.out(0);
        return c;
    }

    private static boolean readsWritten(Chain c, Set<Storage> written) {
        for (Tensor t : c.inputs) {
            if (written.contains(t.storage())) {
                return true;
            }
        }
        return false;
    }

    /** Inputs of c after absorbing prev (shared tensors counted once). */
    private static int merged(Chain c, Chain prev) {
        int count = c.inputs.size();
        for (Tensor t : prev.inputs) {
            boolean seen = false;
            for (Tensor u : c.inputs) {
                seen |= u.storage() == t.storage() && u.shape().equals(t.shape());
            }
            count += seen ? 0 : 1;
        }
        return count;
    }

    private static int inputIndex(Chain c, Tensor t) {
        for (int i = 0; i < c.inputs.size(); i++) {
            Tensor u = c.inputs.get(i);
            if (u.storage() == t.storage() && u.shape().equals(t.shape())) {
                return i;
            }
        }
        c.inputs.add(t);
        return c.inputs.size() - 1;
    }

    private static int remap(int operand, int[] remapIn, int shift) {
        return operand >= 0 ? remapIn[operand] : -1 - ((-1 - operand) + shift);
    }

    private static Node chainNode(Chain c) {
        int inputs = c.inputs.size();
        Shape out = c.output.rank() == 0 ? Shape.of(1) : c.output.shape();
        int rank = out.rank();
        int[] meta = new int[3 + rank + FusedKernels.MAX_INPUTS * rank + 5 * c.instructions.size()];
        meta[0] = inputs;
        meta[1] = c.instructions.size();
        meta[2] = rank;
        int[] dims = out.dims();
        System.arraycopy(dims, 0, meta, 3, rank);
        for (int k = 0; k < inputs; k++) {
            Tensor t = c.inputs.get(k);
            int[] st = t.rank() == 0 ? new int[rank] : t.shape().broadcastStrides(out);
            System.arraycopy(st, 0, meta, 3 + (1 + k) * rank, rank);
        }
        float[] params = new float[2 * c.instructions.size()];
        int p = 3 + rank + FusedKernels.MAX_INPUTS * rank;
        meta = java.util.Arrays.copyOf(meta, meta.length + 1 + 1 + c.stores.size());
        for (int j = 0; j < c.instructions.size(); j++) {
            int[] ins = c.instructions.get(j);
            meta[p] = ins[0];
            meta[p + 1] = ins[1];
            meta[p + 2] = ins[2] >= 0 ? ins[2] : inputs + (-1 - ins[2]);
            meta[p + 3] = ins[3] >= 0 ? ins[3] : inputs + (-1 - ins[3]);
            meta[p + 4] = inputs + j;
            params[2 * j] = c.params.get(j)[0];
            params[2 * j + 1] = c.params.get(j)[1];
            p += 5;
        }
        // outputs: the chain's result, then the intermediates it stores
        Tensor[] outs = new Tensor[1 + c.stores.size()];
        outs[0] = c.output;
        meta[p] = outs.length;
        meta[p + 1] = inputs + c.instructions.size() - 1;
        for (int k = 0; k < c.stores.size(); k++) {
            outs[1 + k] = c.stores.get(k);
            meta[p + 2 + k] = inputs + c.storeInstr.get(k);
        }
        return new Node(ElementwiseFused.OP, c.inputs.toArray(new Tensor[0]), outs, meta, params, true);
    }

    private static boolean fusable(Tensor t, IdentityHashMap<Storage, Integer> uses, Set<Storage> wanted) {
        Storage s = t.storage();
        return !s.isPersistent() && s.isInternal() && uses.getOrDefault(s, 0) == 1 && !wanted.contains(s) && !t.requiresGrad();
    }

    /** The node replacing producer p and consumer c (adding any folding node to {@code folds}), or null. */
    private static Node fuse(Node p, Node c, IdentityHashMap<Storage, Integer> producer, List<Node> folds) {
        if (p.op == Conv2d.OP && c.op == BatchNorm2d.OP) {
            return convBatchNorm(p, c, producer, folds);
        }
        if (p.op == Linear.OP && c.op == Unary.OP && c.iattr[0] == Unary.Kind.GELU.code && p.iattr[4] == Linear.NONE && p.out(0).dtype() == c.out(0).dtype()) {
            int[] a = p.iattr.clone();
            a[4] = Linear.GELU;
            return new Node(Linear.OP, p.inputs, c.outputs, a, p.fattr, true);
        }
        if (p.op == Linear.OP && c.op == Unary.OP && c.iattr[0] == Unary.Kind.GELU_TANH.code && p.iattr[4] == Linear.NONE && p.out(0).dtype() == DType.F16
                && c.out(0).dtype() == DType.F16) {
            // tanh GELU in the epilogue of an FP16 linear layer (the FP16 epilogues take it)
            int[] a = p.iattr.clone();
            a[4] = Linear.GELU_TANH;
            return new Node(Linear.OP, p.inputs, c.outputs, a, p.fattr, true);
        }
        if (p.op == Binary.OP && p.iattr[0] == Binary.Kind.ADD.code && c.op == RowNorm.OP && p.out(0).dtype() == DType.F16 && c.out(0).dtype() == DType.F16
                && c.iattr.length == 4 && c.in(0).storage() == p.out(0).storage() && p.in(0).dtype() == DType.F16 && p.in(1).dtype() == DType.F16) {
            // half-precision inference: LayerNorm(a + b) reads a and b, the sum never stored
            Tensor[] in = java.util.Arrays.copyOf(c.inputs, c.inputs.length + 1);
            in[0] = p.in(0);
            in[c.inputs.length] = p.in(1);
            int[] a = java.util.Arrays.copyOf(c.iattr, 5);
            a[4] = 1;
            return new Node(RowNorm.OP, in, c.outputs, a, c.fattr, true);
        }
        if (p.op == Binary.OP && p.iattr[0] == Binary.Kind.ADD.code && c.op == Unary.OP && c.iattr[0] == Unary.Kind.RELU.code) {
            int[] a = p.iattr.clone();
            a[0] = Binary.Kind.ADD_RELU.code;
            return new Node(Binary.OP, p.inputs, c.outputs, a, p.fattr, true);
        }
        return null;
    }

    private static Node convBatchNorm(Node conv, Node bn, IdentityHashMap<Storage, Integer> producer, List<Node> folds) {
        int[] a = conv.iattr;
        // eval-mode BatchNorm on the convolution's whole output, no ReLU between them
        if (bn.iattr[3] != 0 || a[13] != 0 || bn.in(0).storage() != conv.out(0).storage()) {
            return null;
        }
        Tensor w = conv.in(1);
        Tensor bias = a[12] != 0 ? conv.in(2) : null;
        Tensor[] sources = bias == null ? new Tensor[] { w, bn.in(1), bn.in(2), bn.in(3), bn.in(4) } : new Tensor[] { w, bias, bn.in(1), bn.in(2), bn.in(3), bn.in(4) };
        List<Long> key = new ArrayList<>();
        for (Tensor t : sources) {
            Storage s = t.storage();
            // folding needs constants: persistent tensors that this trace does not write
            if (!s.isPersistent() || producer.containsKey(s)) {
                return null;
            }
            key.add(s.id());
        }
        key.add((long) Float.floatToRawIntBits(bn.fattr[0]));
        Folded f = FOLDED.get(key);
        if (f == null || !f.current()) {
            if (f == null) {
                // host arrays: nothing may be recorded while the executor runs
                f = new Folded(Tensor.persistent(Tensor.of(new float[w.numel()], w.shape().dims())), Tensor.persistent(Tensor.of(new float[a[4]], a[4])),
                        new Storage[sources.length], new long[sources.length]);
                FOLDED.put(key, f);
            }
            for (int i = 0; i < sources.length; i++) {
                f.sources()[i] = sources[i].storage();
                f.versions()[i] = sources[i].storage().version();
            }
            Tensor[] ins = bias == null ? new Tensor[] { w, bn.in(1), bn.in(2), bn.in(3), bn.in(4) } : new Tensor[] { w, bias, bn.in(1), bn.in(2), bn.in(3), bn.in(4) };
            folds.add(new Node(FOLD_BN, ins, new Tensor[] { f.w(), f.b() }, new int[] { a[4], w.numel() / a[4], bias == null ? 0 : 1 }, new float[] { bn.fattr[0] }, true));
        }
        int[] ca = a.clone();
        ca[12] = 1;
        ca[13] = bn.iattr[4];
        return new Node(Conv2d.OP, new Tensor[] { conv.in(0), f.w(), f.b() }, bn.outputs, ca, conv.fattr, true);
    }

    /** inputs [w, (bias), gamma, beta, mean, var]; outputs [w', b']; iattr [cout, per, hasBias]; fattr [eps]. */
    static final class FoldBatchNorm implements Op {

        @Override
        public String name() {
            return "fold_batch_norm";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int cout = n.iattr[0];
            int per = n.iattr[1];
            int hasBias = n.iattr[2];
            int o = hasBias;
            FloatArray w = e.f32(n.in(0));
            FloatArray bias = hasBias != 0 ? e.f32(n.in(1)) : e.dummyF32();
            FloatArray gamma = e.f32(n.in(1 + o));
            FloatArray beta = e.f32(n.in(2 + o));
            FloatArray mean = e.f32(n.in(3 + o));
            FloatArray var = e.f32(n.in(4 + o));
            FloatArray wOut = e.f32(n.out(0));
            FloatArray bOut = e.f32(n.out(1));
            float eps = n.fattr[0];
            e.task(Emitter.grid1D(cout * per), (g, id) -> g.task(id, FoldKernels::batchNorm, e.context(), w, bias, gamma, beta, mean, var, wOut, bOut, cout, per, hasBias, eps));
        }

        @Override
        public void reference(Node n) {
            int cout = n.iattr[0];
            int per = n.iattr[1];
            int o = n.iattr[2];
            FloatArray w = (FloatArray) n.in(0).storage().host();
            FloatArray bias = o != 0 ? (FloatArray) n.in(1).storage().host() : null;
            FloatArray gamma = (FloatArray) n.in(1 + o).storage().host();
            FloatArray beta = (FloatArray) n.in(2 + o).storage().host();
            FloatArray mean = (FloatArray) n.in(3 + o).storage().host();
            FloatArray var = (FloatArray) n.in(4 + o).storage().host();
            FloatArray wOut = (FloatArray) n.out(0).storage().host();
            FloatArray bOut = (FloatArray) n.out(1).storage().host();
            for (int co = 0; co < cout; co++) {
                double s = gamma.get(co) / Math.sqrt((double) var.get(co) + n.fattr[0]);
                for (int j = 0; j < per; j++) {
                    wOut.set(co * per + j, (float) (w.get(co * per + j) * s));
                }
                double b = bias == null ? 0 : bias.get(co);
                bOut.set(co, (float) (beta.get(co) + (b - mean.get(co)) * s));
            }
        }
    }
}
