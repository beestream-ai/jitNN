package org.beehive.jitnn.ops;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.engine.Storage;
import org.beehive.jitnn.kernels.GemmKernels;
import org.beehive.jitnn.kernels.LibraryGlueKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Channels-last inference for convolutions a library runs channels-last (MLX on Metal). Tensors keep
 * their NCHW shape, but those that only convolutions and same-shape element-wise operators read and
 * write are stored [N, H, W, C]: the convolution reads and writes them directly, the element-wise
 * operators do not care, and a layout conversion runs only where an operator needs NCHW (pooling,
 * convolutions on jitNN's kernels, outputs). Without the pass, every library convolution transposes
 * its input and output. {@code -Djitnn.channelsLast=false} turns it off.
 */
public final class ChannelsLast {

    private static final boolean ENABLED = !"false".equals(System.getProperty("jitnn.channelsLast"));
    /** Logs each layout conversion and the operator that needs it ({@code -Djitnn.channelsLast.verbose=true}). */
    private static final boolean VERBOSE = Boolean.getBoolean("jitnn.channelsLast.verbose");

    static final Conv CONV = new Conv();
    static final Winograd WINOGRAD = new Winograd();
    static final Layout TO_HALF = new Layout(2);
    static final AddBias ADD_BIAS = new AddBias();

    /** iattr index (after Conv2d's) set to 1 when the convolution's weight is a channels-last copy. */
    private static final int WEIGHT_LAST = 14;

    /** Winograd iattr index (after WEIGHT_LAST) set when the input still needs its producer's bias (last input) and, at PRE + 1, ReLU. */
    private static final int PRE = 15;

    /** A weight's channels-last copy and the version of the weight it was made from. */
    private record LastWeight(Tensor w, long version) {
    }

    /** Channels-last copies of persistent convolution weights, by the weight's storage. */
    private static final java.util.Map<Storage, LastWeight> WEIGHTS = java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());
    static final Layout TO_LAST = new Layout(1);
    static final Layout TO_FIRST = new Layout(0);

    private static final int N = 0, CIN = 1, H = 2, W = 3, COUT = 4, OH = 5, OW = 6, K = 7, STRIDE = 8, PAD = 9, DIL = 10, GROUPS = 11, BIAS = 12, RELU = 13;

    private ChannelsLast() {
    }

    /** Whether a library takes this convolution channels-last on the current device. */
    private static boolean eligible(Node n) {
        if (n.op != Conv2d.OP) {
            return false;
        }
        int[] a = n.iattr;
        DType t = n.out(0).dtype();
        // an FP16 convolution may read an FP32 input: its conversion to channels-last rounds it
        boolean input = n.in(0).dtype() == t || (t == DType.F16 && n.in(0).dtype() == DType.F32);
        if (t != DType.F32 && t != DType.F16 || !input || n.in(1).dtype() != t) {
            return false;
        }
        return org.beehive.jitnn.natives.Libs.mlxConvChannelsLast(a[N], a[CIN], a[H], a[W], a[COUT], a[K], a[STRIDE], a[OH], a[OW], a[GROUPS], a[DIL], t == DType.F16);
    }

    /**
     * A convolution jitNN runs as Winograd F(4x4, 3x3) that MLX does not take (the 3x3 stride-1 shapes
     * MLX would run with Winograd): its transforms run channels-last too, so no conversion surrounds it.
     */
    private static boolean winograd(Node n) {
        if (n.op != Conv2d.OP) {
            return false;
        }
        int[] a = n.iattr;
        DType t = n.out(0).dtype();
        boolean types = (t == DType.F32 || t == DType.F16) && (n.in(0).dtype() == DType.F32 || n.in(0).dtype() == DType.F16)
                && (n.in(1).dtype() == DType.F32 || n.in(1).dtype() == DType.F16);
        return types && Conv2d.winograd(a, a[OH], a[OW]) && Conv2d.winograd4(a[N], a[OH], a[OW]);
    }

    /** A same-shape element-wise operator or cast over rank-4 tensors: indifferent to the layout. */
    private static boolean elementwise(Node n) {
        if ((n.op != Unary.OP && n.op != Binary.OP && n.op != Cast.OP) || n.outputs.length != 1 || n.out(0).rank() != 4) {
            return false;
        }
        for (Tensor t : n.inputs) {
            // a cast changes the type, not the order of the elements
            if (!t.shape().equals(n.out(0).shape()) || (n.op != Cast.OP && t.dtype() != n.out(0).dtype())) {
                return false;
            }
        }
        return true;
    }

    private static boolean pinned(Tensor t, Set<Storage> wanted) {
        return wanted.contains(t.storage()) || t.storage().isPersistent();
    }

    /** The nodes with channels-last storage where it pays; the nodes unchanged when no convolution qualifies. */
    static List<Node> rewrite(List<Node> nodes, Set<Storage> wanted) {
        if (!ENABLED) {
            return nodes;
        }
        boolean any = false;
        for (Node n : nodes) {
            if (eligible(n)) {
                any = true;
                break;
            }
        }
        if (!any) {
            return nodes;
        }
        // storages some operator of this trace writes (a weight folded in this trace is copied again)
        Set<Storage> written = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (Node n : nodes) {
            for (Tensor t : n.outputs) {
                written.add(t.storage());
            }
        }
        // storages holding [N, H, W, C] data, and the converted copies made so far
        Set<Storage> last = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        IdentityHashMap<Storage, Tensor> asLast = new IdentityHashMap<>();
        IdentityHashMap<Storage, Tensor> asFirst = new IdentityHashMap<>();
        IdentityHashMap<Storage, Tensor> asLastHalf = new IdentityHashMap<>();
        IdentityHashMap<Storage, Tensor> castHalf = new IdentityHashMap<>();
        // [N, C, H W] of each channels-last storage (readers may see it through views of other shapes)
        IdentityHashMap<Storage, int[]> dims = new IdentityHashMap<>();
        List<Node> out = new ArrayList<>(nodes.size() + 16);
        for (Node n : nodes) {
            if (eligible(n)) {
                int[] a = n.iattr;
                Tensor x = n.in(0);
                int[] xd = { a[N], a[CIN], a[H] * a[W] };
                Tensor xl;
                if (x.dtype() != n.out(0).dtype() && last.contains(x.storage())) {
                    xl = castToHalf(x, xd, castHalf, out);
                } else if (x.dtype() != n.out(0).dtype()) {
                    xl = convertToHalf(x, xd, asLastHalf, out);
                } else {
                    xl = last.contains(x.storage()) ? x : convert(x, xd, true, asLast, out);
                }
                Tensor y = n.out(0);
                int[] yd = { a[N], a[COUT], a[OH] * a[OW] };
                Tensor[] ins = n.inputs.clone();
                ins[0] = xl;
                int[] attr = java.util.Arrays.copyOf(n.iattr, WEIGHT_LAST + 1);
                Tensor wl = lastWeight(n.in(1), a, written, out);
                if (wl != null) {
                    ins[1] = wl;
                    attr[WEIGHT_LAST] = 1;
                }
                if (pinned(y, wanted)) {
                    Tensor yl = Tensor.result(y.shape(), y.dtype());
                    out.add(new Node(CONV, ins, new Tensor[] { yl }, attr, n.fattr, true));
                    out.add(new Node(TO_FIRST, new Tensor[] { yl }, new Tensor[] { y }, yd, null, true));
                } else {
                    out.add(new Node(CONV, ins, n.outputs, attr, n.fattr, true));
                    last.add(y.storage());
                    dims.put(y.storage(), yd);
                }
                continue;
            }
            if (winograd(n)) {
                int[] a = n.iattr;
                Tensor x = n.in(0);
                Tensor[] ins = n.inputs.clone();
                ins[0] = last.contains(x.storage()) ? x : convert(x, new int[] { a[N], a[CIN], a[H] * a[W] }, true, asLast, out);
                Tensor y = n.out(0);
                int[] yd = { a[N], a[COUT], a[OH] * a[OW] };
                if (pinned(y, wanted)) {
                    Tensor yl = Tensor.result(y.shape(), y.dtype());
                    out.add(new Node(WINOGRAD, ins, new Tensor[] { yl }, n.iattr, n.fattr, true));
                    out.add(new Node(TO_FIRST, new Tensor[] { yl }, new Tensor[] { y }, yd, null, true));
                } else {
                    out.add(new Node(WINOGRAD, ins, n.outputs, n.iattr, n.fattr, true));
                    last.add(y.storage());
                    dims.put(y.storage(), yd);
                }
                continue;
            }
            boolean anyLast = false;
            for (Tensor t : n.inputs) {
                anyLast |= last.contains(t.storage());
            }
            if (!anyLast) {
                out.add(n);
                continue;
            }
            if (elementwise(n) && !pinned(n.out(0), wanted)) {
                // the operator runs on [N, H, W, C] data: its other inputs are converted to it
                Tensor[] ins = n.inputs.clone();
                for (int i = 0; i < ins.length; i++) {
                    if (!last.contains(ins[i].storage())) {
                        ins[i] = convert(ins[i], layoutAttr(ins[i]), true, asLast, out);
                    }
                }
                out.add(new Node(n.op, ins, n.outputs, n.iattr, n.fattr, true));
                last.add(n.out(0).storage());
                dims.put(n.out(0).storage(), layoutAttr(n.out(0)));
                continue;
            }
            // an operator that needs NCHW reads converted copies
            Tensor[] ins = n.inputs.clone();
            for (int i = 0; i < ins.length; i++) {
                if (last.contains(ins[i].storage())) {
                    if (VERBOSE && !asFirst.containsKey(ins[i].storage())) {
                        System.err.printf("[jitnn] channels-last: %s reads %s in NCHW%n", n, ins[i].shape());
                    }
                    ins[i] = convert(ins[i], dims.get(ins[i].storage()), false, asFirst, out);
                }
            }
            out.add(new Node(n.op, ins, n.outputs, n.iattr, n.fattr, true));
        }
        return deferEpilogues(biasIntoAdd(castsIntoWinograd(out, wanted), wanted), wanted);
    }

    /**
     * A cast from FP16 whose only reader is a Winograd convolution goes away: the input transform reads
     * the FP16 tensor itself (in FP32 arithmetic), and the producer's bias can then move into it too.
     */
    private static List<Node> castsIntoWinograd(List<Node> nodes, Set<Storage> wanted) {
        IdentityHashMap<Storage, Integer> uses = new IdentityHashMap<>();
        IdentityHashMap<Storage, Integer> reader = new IdentityHashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (Tensor t : nodes.get(i).inputs) {
                uses.merge(t.storage(), 1, Integer::sum);
                reader.put(t.storage(), i);
            }
        }
        List<Node> out = new ArrayList<>(nodes);
        boolean dropped = false;
        for (int i = 0; i < out.size(); i++) {
            Node c = out.get(i);
            if (c == null || c.op != Cast.OP || c.inputs.length != 1 || c.in(0).dtype() != DType.F16 || c.out(0).dtype() != DType.F32 || pinned(c.out(0), wanted)) {
                continue;
            }
            Storage y = c.out(0).storage();
            Integer j = reader.get(y);
            if (j == null || j <= i || uses.getOrDefault(y, 0) != 1 || out.get(j) == null || out.get(j).op != WINOGRAD || out.get(j).in(0).storage() != y) {
                continue;
            }
            Node r = out.get(j);
            Tensor[] rin = r.inputs.clone();
            rin[0] = c.in(0);
            out.set(j, new Node(WINOGRAD, rin, r.outputs, r.iattr, r.fattr, true));
            out.set(i, null);
            dropped = true;
        }
        if (!dropped) {
            return out;
        }
        List<Node> kept = new ArrayList<>(out.size());
        for (Node n : out) {
            if (n != null) {
                kept.add(n);
            }
        }
        return kept;
    }

    /**
     * The bias and ReLU of a channels-last MLX convolution moved into its only reader, so no pass of
     * their own runs: a residual add that already takes one convolution's bias takes this one's as a
     * second, the input transform of a Winograd convolution applies them as it reads, and a conversion
     * back to NCHW applies them as it writes.
     */
    private static List<Node> deferEpilogues(List<Node> nodes, Set<Storage> wanted) {
        IdentityHashMap<Storage, Integer> uses = new IdentityHashMap<>();
        IdentityHashMap<Storage, Integer> reader = new IdentityHashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (Tensor t : nodes.get(i).inputs) {
                uses.merge(t.storage(), 1, Integer::sum);
                reader.put(t.storage(), i);
            }
        }
        List<Node> out = new ArrayList<>(nodes);
        for (int i = 0; i < out.size(); i++) {
            Node c = out.get(i);
            if (c.op != CONV || c.iattr[BIAS] == 0 && c.iattr[RELU] == 0 || pinned(c.out(0), wanted)) {
                continue;
            }
            Storage y = c.out(0).storage();
            Integer j = reader.get(y);
            if (j == null || j <= i || uses.getOrDefault(y, 0) != 1) {
                continue;
            }
            Node r = out.get(j);
            boolean hasBias = c.iattr[BIAS] != 0;
            Tensor bias = hasBias ? c.in(2) : null;
            Node replaced = null;
            if (r.op == ADD_BIAS && r.inputs.length == 3 && hasBias && c.iattr[RELU] == 0 && r.in(1).storage() == y) {
                replaced = new Node(ADD_BIAS, new Tensor[] { r.in(0), r.in(1), r.in(2), bias }, r.outputs, r.iattr, r.fattr, true);
            } else if (r.op == WINOGRAD && hasBias && r.in(0).storage() == y && !(r.iattr.length > PRE && r.iattr[PRE] != 0)) {
                int[] ra = java.util.Arrays.copyOf(r.iattr, PRE + 2);
                ra[PRE] = 1;
                ra[PRE + 1] = c.iattr[RELU];
                Tensor[] rin = java.util.Arrays.copyOf(r.inputs, r.inputs.length + 1);
                rin[rin.length - 1] = bias;
                replaced = new Node(WINOGRAD, rin, r.outputs, ra, r.fattr, true);
            } else if (r.op == TO_FIRST && r.inputs.length == 1 && r.in(0).storage() == y) {
                int ep = (hasBias ? GemmKernels.BIAS : 0) | (c.iattr[RELU] != 0 ? GemmKernels.RELU : 0);
                int[] ra = java.util.Arrays.copyOf(r.iattr, 4);
                ra[3] = ep;
                Tensor[] rin = hasBias ? new Tensor[] { r.in(0), bias } : new Tensor[] { r.in(0) };
                replaced = new Node(TO_FIRST, rin, r.outputs, ra, r.fattr, true);
            }
            if (replaced == null) {
                continue;
            }
            int[] ca = c.iattr.clone();
            ca[BIAS] = 0;
            ca[RELU] = 0;
            out.set(i, new Node(CONV, new Tensor[] { c.in(0), c.in(1) }, c.outputs, ca, c.fattr, true));
            out.set(j, replaced);
        }
        return out;
    }

    /**
     * The persistent channels-last copy [Cout, K, K, Cin] of a persistent convolution weight, with a
     * conversion node when it is new, stale (the weight's version moved) or the weight is written in
     * this trace; null for weights that are not persistent (converted on every call instead).
     */
    private static Tensor lastWeight(Tensor w, int[] a, Set<Storage> written, List<Node> out) {
        Storage s = w.storage();
        if (!s.isPersistent() || w.rank() != 4) {
            return null;
        }
        LastWeight c = WEIGHTS.get(s);
        boolean stale = c == null || c.version() != s.version() || written.contains(s);
        if (c == null) {
            int[] dims = { a[COUT], a[K], a[K], a[CIN] };
            float[] zeros = new float[w.numel()];
            Tensor t = w.dtype() == DType.F16 ? Tensor.persistentHalf(zeros, dims) : Tensor.persistent(Tensor.of(zeros, dims));
            c = new LastWeight(t, s.version());
        }
        if (stale) {
            out.add(new Node(TO_LAST, new Tensor[] { w }, new Tensor[] { c.w() }, new int[] { a[COUT], a[CIN], a[K] * a[K] }, null, true));
            WEIGHTS.put(s, new LastWeight(c.w(), s.version()));
        }
        return c.w();
    }

    /**
     * A channels-last convolution whose only reader is a same-shape add (or add + ReLU) leaves its bias
     * to the add: one pass applies bias, residual and ReLU instead of two.
     */
    private static List<Node> biasIntoAdd(List<Node> nodes, Set<Storage> wanted) {
        IdentityHashMap<Storage, Integer> uses = new IdentityHashMap<>();
        IdentityHashMap<Storage, Integer> reader = new IdentityHashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (Tensor t : nodes.get(i).inputs) {
                uses.merge(t.storage(), 1, Integer::sum);
                reader.put(t.storage(), i);
            }
        }
        List<Node> out = new ArrayList<>(nodes);
        for (int i = 0; i < out.size(); i++) {
            Node c = out.get(i);
            if (c.op != CONV || c.iattr[BIAS] == 0 || c.iattr[RELU] != 0 || pinned(c.out(0), wanted)) {
                continue;
            }
            Storage y = c.out(0).storage();
            Integer j = reader.get(y);
            if (j == null || j <= i || uses.getOrDefault(y, 0) != 1) {
                continue;
            }
            Node add = out.get(j);
            boolean addOp = add.op == Binary.OP && add.inputs.length == 2
                    && (add.iattr[0] == Binary.Kind.ADD.code || add.iattr[0] == Binary.Kind.ADD_RELU.code) && add.in(0).shape().equals(add.in(1).shape())
                    && add.in(0).dtype() == add.in(1).dtype() && add.out(0).dtype() == add.in(0).dtype() && add.in(0).dtype() == c.out(0).dtype();
            if (!addOp) {
                continue;
            }
            Tensor other = add.in(0).storage() == y ? add.in(1) : add.in(0);
            int[] ca = c.iattr.clone();
            ca[BIAS] = 0;
            Tensor bias = c.in(2);
            out.set(i, new Node(CONV, new Tensor[] { c.in(0), c.in(1) }, c.outputs, ca, c.fattr, true));
            int relu = add.iattr[0] == Binary.Kind.ADD_RELU.code ? 1 : 0;
            out.set(j, new Node(ADD_BIAS, new Tensor[] { c.out(0), other, bias }, add.outputs, new int[] { c.iattr[COUT], relu }, null, true));
        }
        return out;
    }

    private static int[] layoutAttr(Tensor t) {
        return new int[] { t.dim(0), t.dim(1), t.dim(2) * t.dim(3) };
    }

    /**
     * t in the other layout, of NCHW extents nchw = [N, C, H W]: one conversion per storage and
     * direction, reused by every later reader (the copy keeps t's shape, which may be a view's).
     */
    private static Tensor convert(Tensor t, int[] nchw, boolean toLast, IdentityHashMap<Storage, Tensor> cache, List<Node> out) {
        Tensor c = cache.get(t.storage());
        if (c == null) {
            if (VERBOSE) {
                System.err.printf("[jitnn] channels-last: %s %s %s%n", toLast ? "to NHWC" : "to NCHW", t.dtype(), java.util.Arrays.toString(nchw));
            }
            c = Tensor.result(t.shape(), t.dtype());
            out.add(new Node(toLast ? TO_LAST : TO_FIRST, new Tensor[] { t }, new Tensor[] { c }, nchw, null, true));
            cache.put(t.storage(), c);
        }
        return c;
    }

    /** An FP32 channels-last tensor as FP16 channels-last (nchw = [N, C, H W]), once per storage. */
    private static Tensor castToHalf(Tensor t, int[] nchw, IdentityHashMap<Storage, Tensor> cache, List<Node> out) {
        Tensor c = cache.get(t.storage());
        if (c == null) {
            c = Tensor.result(t.shape(), DType.F16);
            out.add(new Node(TO_HALF, new Tensor[] { t }, new Tensor[] { c }, nchw, null, true));
            cache.put(t.storage(), c);
        }
        return c;
    }

    /** An FP32 NCHW tensor as FP16 NHWC (nchw = [N, C, H W]), once per storage. */
    private static Tensor convertToHalf(Tensor t, int[] nchw, IdentityHashMap<Storage, Tensor> cache, List<Node> out) {
        Tensor c = cache.get(t.storage());
        if (c == null) {
            c = Tensor.result(t.shape(), DType.F16);
            out.add(new Node(TO_LAST, new Tensor[] { t }, new Tensor[] { c }, nchw, null, true));
            cache.put(t.storage(), c);
        }
        return c;
    }

    /**
     * NCHW to NHWC (mode 1, also from FP32 into FP16) or back (mode 0), or FP32 to FP16 in the same
     * layout (mode 2); iattr [N, C, H W].
     */
    static final class Layout implements Op {

        private final int mode;
        private final boolean toLast;

        Layout(int mode) {
            this.mode = mode;
            this.toLast = mode == 1;
        }

        @Override
        public String name() {
            return mode == 2 ? "to_half" : toLast ? "to_channels_last" : "to_channels_first";
        }

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node n) {
            int c = n.iattr[1];
            int hw = n.iattr[2];
            int total = n.iattr[0] * c * hw;
            // back to NCHW may also apply the producing convolution's bias and ReLU (iattr[3], input 1)
            int ep = n.iattr.length > 3 ? n.iattr[3] : 0;
            if (mode == 2) {
                FloatArray x = e.f32(n.in(0));
                HalfFloatArray y = e.f16(n.out(0));
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), x, y, total));
                return;
            }
            if (n.out(0).dtype() == DType.F16 && n.in(0).dtype() == DType.F32) {
                FloatArray x = e.f32(n.in(0));
                HalfFloatArray y = e.f16(n.out(0));
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLastToHalf, e.context(), x, y, c, hw, total));
                return;
            }
            if (n.out(0).dtype() == DType.F16) {
                HalfFloatArray x = e.f16(n.in(0));
                HalfFloatArray y = e.f16(n.out(0));
                if (toLast) {
                    e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLastHalf, e.context(), x, y, c, hw, total));
                } else {
                    FloatArray b = n.inputs.length > 1 ? e.f32(n.in(1)) : e.dummyF32();
                    e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::fromChannelsLastHalf, e.context(), x, y, b, c, hw, total, ep));
                }
                return;
            }
            FloatArray x = e.f32(n.in(0));
            FloatArray y = e.f32(n.out(0));
            if (toLast) {
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLast, e.context(), x, y, c, hw, total));
            } else {
                FloatArray b = n.inputs.length > 1 ? e.f32(n.in(1)) : e.dummyF32();
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::fromChannelsLast, e.context(), x, y, b, c, hw, total, ep));
            }
        }

        @Override
        public void reference(Node n) {
            throw new UnsupportedOperationException(name() + " runs only in compiled plans");
        }
    }

    /** {@link Conv2d} on [N, H, W, C] input and output (iattr as Conv2d's); the weight stays [Cout, Cin, K, K]. */
    static final class Conv implements Op {

        @Override
        public String name() {
            return "conv2d_channels_last";
        }

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            int epilogue = (a[BIAS] != 0 ? GemmKernels.BIAS : 0) | (a[RELU] != 0 ? GemmKernels.RELU : 0);
            FloatArray bias = a[BIAS] != 0 ? e.f32(node.in(2)) : null;
            boolean weightLast = a.length > WEIGHT_LAST && a[WEIGHT_LAST] != 0;
            if (node.out(0).dtype() == DType.F16) {
                org.beehive.jitnn.natives.Libs.conv2dChannelsLastHalf(e, e.f16(node.in(0)), e.f16(node.in(1)), e.f16(node.out(0)), bias, a[N], a[CIN], a[H], a[W], a[COUT], a[OH],
                        a[OW], a[K], a[STRIDE], a[PAD], epilogue, weightLast);
            } else {
                org.beehive.jitnn.natives.Libs.conv2dChannelsLast(e, e.f32(node.in(0)), e.f32(node.in(1)), e.f32(node.out(0)), bias, a[N], a[CIN], a[H], a[W], a[COUT], a[OH],
                        a[OW], a[K], a[STRIDE], a[PAD], epilogue, weightLast);
            }
        }

        @Override
        public void reference(Node n) {
            throw new UnsupportedOperationException(name() + " runs only in compiled plans");
        }
    }

    /**
     * {@link Conv2d} as Winograd F(4x4, 3x3) on [N, H, W, C] input and output (iattr as Conv2d's):
     * channels-last input and output transforms around the 36 batched products (MLX where it takes
     * them, jitNN's GEMM otherwise).
     */
    static final class Winograd implements Op {

        @Override
        public String name() {
            return "conv2d_winograd_channels_last";
        }

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            int cin = a[CIN], h = a[H], wd = a[W], cout = a[COUT], oh = a[OH], ow = a[OW], pad = a[PAD];
            int th = Emitter.ceilDiv(oh, 4);
            int tw = Emitter.ceilDiv(ow, 4);
            int tiles = a[N] * th * tw;
            FloatArray v = e.sharedScratch("winogradV", 36 * cin * tiles);
            FloatArray u = e.sharedScratch("winogradU", 36 * cout * cin);
            FloatArray mm = e.sharedScratch("winogradM", 36 * cout * tiles);
            boolean pre = a.length > PRE && a[PRE] != 0;
            if (pre) {
                // the producing convolution's bias and ReLU, applied as the inputs are read
                FloatArray preBias = e.f32(node.in(node.inputs.length - 1));
                int preRelu = a[PRE + 1];
                if (node.in(0).dtype() == DType.F16) {
                    HalfFloatArray x = e.f16(node.in(0));
                    e.task(Emitter.grid1D(cin * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4InputLastActHalf, e.context(), x, preBias,
                            preRelu, v, cin, h, wd, pad, th, tw, tiles));
                } else {
                    FloatArray x = e.f32(node.in(0));
                    e.task(Emitter.grid1D(cin * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4InputLastAct, e.context(), x, preBias,
                            preRelu, v, cin, h, wd, pad, th, tw, tiles));
                }
            } else if (node.in(0).dtype() == DType.F16) {
                HalfFloatArray x = e.f16(node.in(0));
                e.task(Emitter.grid1D(cin * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4InputLastHalf, e.context(), x, v, cin, h, wd,
                        pad, th, tw, tiles));
            } else {
                FloatArray x = e.f32(node.in(0));
                e.task(Emitter.grid1D(cin * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4InputLast, e.context(), x, v, cin, h, wd, pad,
                        th, tw, tiles));
            }
            FloatArray w;
            if (node.in(1).dtype() == DType.F16) {
                HalfFloatArray wh = e.f16(node.in(1));
                int nw = cout * cin * 9;
                w = e.sharedScratch("winogradW", nw);
                FloatArray wf = w;
                e.task(Emitter.grid1D(nw), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f16ToF32, e.context(), wh, wf, nw));
            } else {
                w = e.f32(node.in(1));
            }
            FloatArray wt = w;
            e.task(Emitter.grid1D(cout * cin), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4WeightLast, e.context(), wt, u, cout, cin, 0));
            // mm[k] [tiles, cout] = v[k] [tiles, cin] u[k] [cin, cout] for the 36 tile positions k
            int tilesY = Emitter.ceilDiv(tiles, GemmKernels.BM);
            FloatArray none = e.dummyF32();
            if (!org.beehive.jitnn.natives.Libs.gemmBatched(e, v, u, mm, null, tiles, cout, cin, 0, 0, 0, 36, tiles * cin, cin * cout)) {
                e.task(Emitter.blocks2D(Emitter.ceilDiv(cout, GemmKernels.BN), tilesY * 36, 16, 16),
                        (g, id) -> g.task(id, GemmKernels::gemm64pdBatched, e.context(), v, u, mm, none, tiles, cout, cin, 0, 0, 0, tiles * cin, cin * cout, tilesY));
            }
            int epilogue = (a[BIAS] != 0 ? GemmKernels.BIAS : 0) | (a[RELU] != 0 ? GemmKernels.RELU : 0);
            FloatArray bias = a[BIAS] != 0 ? e.f32(node.in(2)) : none;
            if (node.out(0).dtype() == DType.F16) {
                HalfFloatArray y = e.f16(node.out(0));
                e.task(Emitter.grid1D(cout * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4OutputLastHalf, e.context(), mm, y, bias, cout,
                        oh, ow, th, tw, tiles, epilogue));
            } else {
                FloatArray y = e.f32(node.out(0));
                e.task(Emitter.grid1D(cout * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4OutputLast, e.context(), mm, y, bias, cout, oh,
                        ow, th, tw, tiles, epilogue));
            }
        }

        @Override
        public void reference(Node n) {
            throw new UnsupportedOperationException(name() + " runs only in compiled plans");
        }
    }

    /**
     * out = y + bias[c] + other (+ bias2[c]) (+ ReLU) on channels-last tensors: inputs [y, other, bias
     * (, bias2)], iattr [C, relu].
     */
    static final class AddBias implements Op {

        @Override
        public String name() {
            return "add_bias_channels_last";
        }

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node n) {
            int c = n.iattr[0];
            int relu = n.iattr[1];
            int total = n.out(0).numel();
            FloatArray bias = e.f32(n.in(2));
            int hasBias2 = n.inputs.length > 3 ? 1 : 0;
            FloatArray bias2 = hasBias2 != 0 ? e.f32(n.in(3)) : e.dummyF32();
            if (n.out(0).dtype() == DType.F16) {
                HalfFloatArray y = e.f16(n.in(0));
                HalfFloatArray r = e.f16(n.in(1));
                HalfFloatArray o = e.f16(n.out(0));
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::addBiasLastHalf, e.context(), y, r, bias, bias2, o, c, total, relu, hasBias2));
            } else {
                FloatArray y = e.f32(n.in(0));
                FloatArray r = e.f32(n.in(1));
                FloatArray o = e.f32(n.out(0));
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::addBiasLast, e.context(), y, r, bias, bias2, o, c, total, relu, hasBias2));
            }
        }

        @Override
        public void reference(Node n) {
            throw new UnsupportedOperationException(name() + " runs only in compiled plans");
        }
    }
}
