package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.CompactKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Operators with data-dependent output sizes: nonzero, masked select and unique (see
 * {@link CompactKernels}). The flags and their prefix sums are computed on the device; the size
 * of the result is the one value read back to the host, which ends the current trace (as
 * PyTorch synchronises for these operators). {@link #nonzero(Tensor, int, int)} takes the size
 * up front instead, padding, and needs no read.
 */
public final class Compact {

    static final Scan SCAN = new Scan();
    static final Values VALUES = new Values();
    static final Coords COORDS = new Coords();
    static final Expand EXPAND = new Expand();
    static final RunLengths RUN_LENGTHS = new RunLengths();
    static final Inverse INVERSE = new Inverse();

    private Compact() {
    }

    /** [flags, offsets, block sums, total] of x under the predicate {@code mode}. */
    static Tensor[] scan(Tensor x, int mode) {
        Checks.f32(x);
        int n = x.numel();
        int blocks = Emitter.ceilDiv(Math.max(n, 1), CompactKernels.BLOCK);
        Tensor flags = Tensor.result(Shape.of(n), DType.I32);
        Tensor offsets = Tensor.result(Shape.of(n), DType.I32);
        Tensor sums = Tensor.result(Shape.of(blocks), DType.I32);
        Tensor total = Tensor.result(Shape.of(1), DType.I32);
        Tensor.record(SCAN, new Tensor[] { x }, new Tensor[] { flags, offsets, sums, total }, new int[] { mode }, null, true);
        return new Tensor[] { flags.detach(), offsets.detach(), sums, total.detach() };
    }

    private static int count(Tensor total) {
        return total.toIntArray()[0];
    }

    /** INT32 coordinates [count, rank] of the non-zero elements of x, in row-major order (torch.nonzero). */
    public static Tensor nonzero(Tensor x) {
        Tensor[] s = scan(x, CompactKernels.NONZERO);
        int count = count(s[3]);
        if (count == 0) {
            return Tensor.of(new int[0], 0, x.rank());
        }
        return coords(x, s, count, 0);
    }

    /** As {@link #nonzero(Tensor)} with a fixed {@code size} rows: extra non-zeros dropped, missing rows filled with {@code fill}. */
    public static Tensor nonzero(Tensor x, int size, int fill) {
        return coords(x, scan(x, CompactKernels.NONZERO), size, fill);
    }

    private static Tensor coords(Tensor x, Tensor[] s, int size, int fill) {
        int rank = Math.max(x.rank(), 1);
        int[] dims = x.rank() == 0 ? new int[] { 1 } : x.shape().dims();
        Tensor out = Tensor.result(Shape.of(size, rank), DType.I32);
        Tensor.record(COORDS, new Tensor[] { s[0], s[1], Tensor.of(dims, rank), s[3] }, new Tensor[] { out }, new int[] { size, fill }, null, true);
        return out;
    }

    /** The elements of x where mask (broadcast to x) is not zero, as a vector (torch.masked_select). */
    public static Tensor maskedSelect(Tensor x, Tensor mask) {
        Checks.f32(x);
        Tensor m = mask.expand(x.shape());
        Tensor[] s = scan(m, CompactKernels.NONZERO);
        int count = count(s[3]);
        Tensor out = Tensor.result(Shape.of(count), DType.F32);
        if (count == 0) {
            return out;
        }
        Tensor.record(VALUES, new Tensor[] { x, s[0], s[1] }, new Tensor[] { out }, new int[] { count }, null, true);
        return out;
    }

    /**
     * The sorted distinct values of x (flattened), and optionally the index of each element's value
     * among them (INT32, x's shape) and how often each occurs (INT32): [values, inverse, counts].
     */
    public static Tensor[] unique(Tensor x, boolean returnInverse, boolean returnCounts) {
        Checks.f32(x);
        int n = x.numel();
        Tensor[] sorted = Ordered.sortWithRanks(x.reshape(n), 0, false);
        Tensor[] s = scan(sorted[0], CompactKernels.RUN_START);
        int count = count(s[3]);
        Tensor values = Tensor.result(Shape.of(count), DType.F32);
        Tensor.record(VALUES, new Tensor[] { sorted[0], s[0], s[1] }, new Tensor[] { values }, new int[] { count }, null, true);
        Tensor inverse = null;
        Tensor counts = null;
        if (returnInverse) {
            inverse = Tensor.result(Shape.of(n), DType.I32);
            Tensor.record(INVERSE, new Tensor[] { sorted[2], s[0], s[1] }, new Tensor[] { inverse }, new int[0], null, true);
            inverse = inverse.reshape(x.shape().dims());
        }
        if (returnCounts) {
            counts = Tensor.result(Shape.of(count), DType.I32);
            Tensor.record(RUN_LENGTHS, new Tensor[] { s[0], s[1] }, new Tensor[] { counts }, new int[0], null, true);
        }
        return new Tensor[] { values.detach(), inverse, counts };
    }

    private static int[] ints(Tensor t) {
        IntArray a = (IntArray) t.storage().host();
        int[] v = new int[t.numel()];
        for (int i = 0; i < v.length; i++) {
            v[i] = a.get(i);
        }
        return v;
    }

    /** inputs [x]; outputs [flags, offsets, block sums, total]; iattr = [mode]. */
    static final class Scan implements Op {

        @Override
        public String name() {
            return "compact_scan";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            IntArray flags = e.i32(node.out(0));
            IntArray offsets = e.i32(node.out(1));
            IntArray sums = e.i32(node.out(2));
            IntArray total = e.i32(node.out(3));
            int n = node.in(0).numel(), blocks = node.out(2).numel(), mode = node.iattr[0];
            e.task(Emitter.blocks1D(blocks, CompactKernels.BLOCK), (g, id) -> g.task(id, CompactKernels::blockScan, e.context(), x, flags, offsets, sums, mode, n));
            e.task(Emitter.blocks1D(1, 32), (g, id) -> g.task(id, CompactKernels::scanBlocks, e.context(), sums, total, blocks));
            e.task(Emitter.grid1D(n), (g, id) -> g.task(id, CompactKernels::addBlockOffsets, e.context(), offsets, sums, n));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            IntArray flags = (IntArray) node.out(0).storage().host();
            IntArray offsets = (IntArray) node.out(1).storage().host();
            IntArray total = (IntArray) node.out(3).storage().host();
            int n = node.in(0).numel(), mode = node.iattr[0];
            int run = 0;
            for (int i = 0; i < n; i++) {
                float v = x.get(i);
                int f = mode == CompactKernels.NONZERO ? (v != 0.0f ? 1 : 0) : (i == 0 || v != x.get(i - 1) ? 1 : 0);
                flags.set(i, f);
                offsets.set(i, run);
                run += f;
            }
            total.set(0, run);
        }
    }

    /** inputs [x, flags, offsets]; outputs [compacted]; iattr = [size]. */
    static final class Values implements Op {

        @Override
        public String name() {
            return "compact_values";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            IntArray flags = e.i32(node.in(1));
            IntArray offsets = e.i32(node.in(2));
            FloatArray out = e.f32(node.out(0));
            int n = node.in(0).numel(), size = node.iattr[0];
            e.task(Emitter.grid1D(n), (g, id) -> g.task(id, CompactKernels::compactValues, e.context(), x, flags, offsets, out, size, n));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            int[] flags = ints(node.in(1));
            int[] offsets = ints(node.in(2));
            FloatArray out = (FloatArray) node.out(0).storage().host();
            for (int i = 0; i < flags.length; i++) {
                if (flags[i] == 1 && offsets[i] < node.iattr[0]) {
                    out.set(offsets[i], x.get(i));
                }
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor g) {
            Tensor gx = Tensor.result(node.in(0).shape(), DType.F32);
            Tensor.record(EXPAND, new Tensor[] { g, node.in(1), node.in(2) }, new Tensor[] { gx }, new int[0], null, true);
            return new Tensor[] { gx, null, null };
        }
    }

    /** inputs [flags, offsets, dims, total]; outputs [coordinates [size, rank]]; iattr = [size, fill]. */
    static final class Coords implements Op {

        @Override
        public String name() {
            return "compact_coords";
        }

        @Override
        public void emit(Emitter e, Node node) {
            IntArray flags = e.i32(node.in(0));
            IntArray offsets = e.i32(node.in(1));
            IntArray dims = e.i32(node.in(2));
            IntArray total = e.i32(node.in(3));
            IntArray out = e.i32(node.out(0));
            int n = node.in(0).numel(), rank = node.in(2).numel(), size = node.iattr[0], fill = node.iattr[1];
            e.task(Emitter.grid1D(Math.max(n, size)), (g, id) -> g.task(id, CompactKernels::compactCoords, e.context(), flags, offsets, dims, total, out, rank, size, fill, n));
        }

        @Override
        public void reference(Node node) {
            int[] flags = ints(node.in(0));
            int[] offsets = ints(node.in(1));
            int[] dims = ints(node.in(2));
            int count = ints(node.in(3))[0];
            IntArray out = (IntArray) node.out(0).storage().host();
            int rank = dims.length, size = node.iattr[0], fill = node.iattr[1];
            for (int i = 0; i < flags.length; i++) {
                if (flags[i] == 1 && offsets[i] < size) {
                    int rest = i;
                    for (int d = rank - 1; d >= 0; d--) {
                        out.set(offsets[i] * rank + d, rest % dims[d]);
                        rest /= dims[d];
                    }
                }
            }
            for (int r = count; r < size; r++) {
                for (int d = 0; d < rank; d++) {
                    out.set(r * rank + d, fill);
                }
            }
        }
    }

    /** inputs [g, flags, offsets]; outputs [gx]. */
    static final class Expand implements Op {

        @Override
        public String name() {
            return "compact_expand";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray g = e.f32(node.in(0));
            IntArray flags = e.i32(node.in(1));
            IntArray offsets = e.i32(node.in(2));
            FloatArray gx = e.f32(node.out(0));
            int n = node.out(0).numel();
            e.task(Emitter.grid1D(n), (gr, id) -> gr.task(id, CompactKernels::expand, e.context(), g, flags, offsets, gx, n));
        }

        @Override
        public void reference(Node node) {
            FloatArray g = (FloatArray) node.in(0).storage().host();
            int[] flags = ints(node.in(1));
            int[] offsets = ints(node.in(2));
            FloatArray gx = (FloatArray) node.out(0).storage().host();
            for (int i = 0; i < flags.length; i++) {
                gx.set(i, flags[i] == 1 ? g.get(offsets[i]) : 0.0f);
            }
        }
    }

    /** inputs [flags, offsets]; outputs [run lengths]. */
    static final class RunLengths implements Op {

        @Override
        public String name() {
            return "compact_run_lengths";
        }

        @Override
        public void emit(Emitter e, Node node) {
            IntArray flags = e.i32(node.in(0));
            IntArray offsets = e.i32(node.in(1));
            IntArray counts = e.i32(node.out(0));
            int n = node.in(0).numel();
            e.task(Emitter.grid1D(n), (g, id) -> g.task(id, CompactKernels::runLengths, e.context(), flags, offsets, counts, n));
        }

        @Override
        public void reference(Node node) {
            int[] flags = ints(node.in(0));
            int[] offsets = ints(node.in(1));
            IntArray counts = (IntArray) node.out(0).storage().host();
            int start = -1;
            for (int i = 0; i <= flags.length; i++) {
                if (i == flags.length || flags[i] == 1) {
                    if (start >= 0) {
                        counts.set(offsets[start], i - start);
                    }
                    start = i;
                }
            }
        }
    }

    /** inputs [ranks, flags, offsets]; outputs [inverse]. */
    static final class Inverse implements Op {

        @Override
        public String name() {
            return "compact_inverse";
        }

        @Override
        public void emit(Emitter e, Node node) {
            IntArray ranks = e.i32(node.in(0));
            IntArray flags = e.i32(node.in(1));
            IntArray offsets = e.i32(node.in(2));
            IntArray inverse = e.i32(node.out(0));
            int n = node.in(0).numel();
            e.task(Emitter.grid1D(n), (g, id) -> g.task(id, CompactKernels::inverse, e.context(), ranks, flags, offsets, inverse, n));
        }

        @Override
        public void reference(Node node) {
            int[] ranks = ints(node.in(0));
            int[] flags = ints(node.in(1));
            int[] offsets = ints(node.in(2));
            IntArray inverse = (IntArray) node.out(0).storage().host();
            for (int j = 0; j < ranks.length; j++) {
                inverse.set(j, offsets[ranks[j]] + flags[ranks[j]] - 1);
            }
        }
    }
}
