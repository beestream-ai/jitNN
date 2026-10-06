package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.IndexKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** index_select, gather, scatter/scatter-add along an axis, and segment reductions (see {@link IndexKernels}). */
public final class Indexing {

    static final IndexSelect INDEX_SELECT = new IndexSelect();
    static final IndexSelectBackward INDEX_SELECT_BACKWARD = new IndexSelectBackward();
    static final Gather GATHER = new Gather();
    static final Scatter SCATTER = new Scatter();
    static final SegmentReduce SEGMENT = new SegmentReduce();
    static final SegmentReduceBackward SEGMENT_BACKWARD = new SegmentReduceBackward();

    private Indexing() {
    }

    private static void checkIndex(Tensor idx) {
        if (idx.dtype() != DType.I32) {
            throw new IllegalArgumentException("indices must be INT32, got " + idx.dtype());
        }
    }

    /** Elements idx[0 .. m-1] of {@code axis} (torch.index_select); idx is INT32 [m]. */
    public static Tensor indexSelect(Tensor x, int axis, Tensor idx) {
        Checks.f32(x);
        checkIndex(idx);
        int[] l = Ordered.lines(x, axis);
        int m = idx.numel();
        int[] dims = x.shape().dims();
        dims[x.shape().axis(axis)] = m;
        Tensor y = Tensor.result(Shape.of(dims), DType.F32);
        Tensor.record(INDEX_SELECT, new Tensor[] { x, idx }, new Tensor[] { y }, new int[] { l[0], l[1], l[2], m }, null, true);
        return y;
    }

    /** torch.gather: out = x[..., index[...], ...] along {@code axis}; index has x's shape except along the axis. */
    public static Tensor gather(Tensor x, int axis, Tensor index) {
        Checks.f32(x);
        checkIndex(index);
        int ax = x.shape().axis(axis);
        sameExcept(x.shape(), index.shape(), ax);
        int[] l = Ordered.lines(x, axis);
        Tensor y = Tensor.result(index.shape(), DType.F32);
        Tensor.record(GATHER, new Tensor[] { x, index }, new Tensor[] { y }, new int[] { l[0], l[1], l[2], index.dim(ax), ax }, null, true);
        return y;
    }

    /**
     * torch.scatter_add (add) or torch.scatter (replace; duplicates resolve to the last index
     * position, deterministically): base with src added or written at index along {@code axis}.
     */
    public static Tensor scatter(Tensor base, int axis, Tensor index, Tensor src, boolean add) {
        Checks.f32(base, src);
        checkIndex(index);
        int ax = base.shape().axis(axis);
        sameExcept(base.shape(), index.shape(), ax);
        if (!index.shape().equals(src.shape())) {
            throw new IllegalArgumentException("scatter: index " + index.shape() + " and src " + src.shape() + " differ");
        }
        int[] l = Ordered.lines(base, axis);
        Tensor y = Tensor.result(base.shape(), DType.F32);
        Tensor.record(SCATTER, new Tensor[] { base, index, src }, new Tensor[] { y }, new int[] { l[0], l[1], l[2], index.dim(ax), add ? 1 : 0, ax }, null, true);
        return y;
    }

    private static void sameExcept(Shape a, Shape b, int axis) {
        boolean ok = a.rank() == b.rank();
        for (int d = 0; ok && d < a.rank(); d++) {
            ok = d == axis || a.dims()[d] == b.dims()[d];
        }
        if (!ok) {
            throw new IllegalArgumentException("index shape " + b + " must match " + a + " except along axis " + axis);
        }
    }

    /**
     * Reduction of row segments of x [rows, D]: segment b covers rows offsets[b] .. offsets[b+1] - 1
     * (the last runs to the end); mode {@link IndexKernels#SUM}, MEAN or MAX. Gives [segments, D].
     */
    public static Tensor segmentReduce(Tensor x, int[] offsets, int mode) {
        Checks.f32(x);
        Checks.rank(x, 2);
        int rows = x.dim(0);
        int segs = offsets.length;
        int[] starts = offsets.clone();
        int[] ends = new int[segs];
        int[] rowSeg = new int[Math.max(rows, 1)];
        java.util.Arrays.fill(rowSeg, -1);
        for (int b = 0; b < segs; b++) {
            ends[b] = b + 1 < segs ? offsets[b + 1] : rows;
            if (starts[b] < 0 || ends[b] < starts[b] || ends[b] > rows) {
                throw new IllegalArgumentException("segment offsets must be non-decreasing within [0, " + rows + "]: " + java.util.Arrays.toString(offsets));
            }
            for (int r = starts[b]; r < ends[b]; r++) {
                rowSeg[r] = b;
            }
        }
        Tensor y = Tensor.result(Shape.of(segs, x.dim(1)), DType.F32);
        Tensor[] in = { x, Tensor.of(starts, segs), Tensor.of(ends, segs), Tensor.of(rowSeg, rowSeg.length) };
        if (mode == IndexKernels.MAX) {
            Tensor.record(SEGMENT, in, new Tensor[] { y, Tensor.result(y.shape(), DType.I32) }, new int[] { mode }, null, true);
        } else {
            Tensor.record(SEGMENT, in, new Tensor[] { y }, new int[] { mode }, null, true);
        }
        return y;
    }

    /** iattr = [outer, n, inner, m]. */
    static final class IndexSelect implements Op {

        @Override
        public String name() {
            return "index_select";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            IntArray idx = e.i32(node.in(1));
            FloatArray y = e.f32(node.out(0));
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3];
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, IndexKernels::indexSelect, e.context(), x, idx, y, n, m, inner, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            IntArray idx = (IntArray) node.in(1).storage().host();
            FloatArray y = (FloatArray) node.out(0).storage().host();
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3];
            for (int i = 0; i < node.out(0).numel(); i++) {
                int q = i % inner, j = (i / inner) % m, o = i / (inner * m);
                y.set(i, x.get((o * n + idx.get(j)) * inner + q));
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor gy) {
            Tensor dx = Tensor.result(node.in(0).shape(), DType.F32);
            Tensor.record(INDEX_SELECT_BACKWARD, new Tensor[] { gy, node.in(1) }, new Tensor[] { dx }, node.iattr.clone(), null, true);
            return new Tensor[] { dx, null };
        }
    }

    static final class IndexSelectBackward implements Op {

        @Override
        public String name() {
            return "index_select_bwd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray dy = e.f32(node.in(0));
            IntArray idx = e.i32(node.in(1));
            FloatArray dx = e.f32(node.out(0));
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3];
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, IndexKernels::indexSelectBackward, e.context(), dy, idx, dx, n, m, inner, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray dy = (FloatArray) node.in(0).storage().host();
            IntArray idx = (IntArray) node.in(1).storage().host();
            FloatArray dx = (FloatArray) node.out(0).storage().host();
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3];
            double[] acc = new double[node.out(0).numel()];
            for (int i = 0; i < node.in(0).numel(); i++) {
                int q = i % inner, j = (i / inner) % m, o = i / (inner * m);
                acc[(o * n + idx.get(j)) * inner + q] += dy.get(i);
            }
            for (int i = 0; i < acc.length; i++) {
                dx.set(i, (float) acc[i]);
            }
        }
    }

    /** iattr = [outer, n, inner, m, axis]. */
    static final class Gather implements Op {

        @Override
        public String name() {
            return "gather";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            IntArray index = e.i32(node.in(1));
            FloatArray y = e.f32(node.out(0));
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3];
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, IndexKernels::gather, e.context(), x, index, y, n, m, inner, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            IntArray index = (IntArray) node.in(1).storage().host();
            FloatArray y = (FloatArray) node.out(0).storage().host();
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3];
            for (int i = 0; i < node.out(0).numel(); i++) {
                int q = i % inner, o = i / (inner * m);
                y.set(i, x.get((o * n + index.get(i)) * inner + q));
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor gy) {
            Tensor x = node.in(0);
            Tensor dx = scatter(Tensor.zeros(x.shape().dims()), node.iattr[4], node.in(1), gy, true);
            return new Tensor[] { dx, null };
        }
    }

    /** iattr = [outer, n, inner, m, add, axis]. */
    static final class Scatter implements Op {

        @Override
        public String name() {
            return "scatter";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray base = e.f32(node.in(0));
            IntArray index = e.i32(node.in(1));
            FloatArray src = e.f32(node.in(2));
            FloatArray y = e.f32(node.out(0));
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3], add = node.iattr[4];
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, IndexKernels::scatter, e.context(), base, index, src, y, n, m, inner, add, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray base = (FloatArray) node.in(0).storage().host();
            IntArray index = (IntArray) node.in(1).storage().host();
            FloatArray src = (FloatArray) node.in(2).storage().host();
            FloatArray y = (FloatArray) node.out(0).storage().host();
            int n = node.iattr[1], inner = node.iattr[2], m = node.iattr[3];
            boolean add = node.iattr[4] != 0;
            double[] acc = new double[node.out(0).numel()];
            for (int i = 0; i < acc.length; i++) {
                acc[i] = base.get(i);
            }
            for (int j = 0; j < node.in(1).numel(); j++) {
                int q = j % inner, o = j / (inner * m);
                int at = (o * n + index.get(j)) * inner + q;
                acc[at] = add ? acc[at] + src.get(j) : src.get(j);
            }
            for (int i = 0; i < acc.length; i++) {
                y.set(i, (float) acc[i]);
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor gy) {
            if (node.iattr[4] == 0) {
                throw new UnsupportedOperationException("scatter (replace) has no gradient; use scatter-add");
            }
            int axis = node.iattr[5];
            Tensor gsrc = node.in(2).requiresGrad() ? gather(gy, axis, node.in(1)) : null;
            return new Tensor[] { node.in(0).requiresGrad() ? gy : null, null, gsrc };
        }
    }

    /** inputs [x, starts, ends, rowSeg]; outputs [y] or [y, argmax]; iattr = [mode]. */
    static final class SegmentReduce implements Op {

        @Override
        public String name() {
            return "segment_reduce";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            IntArray starts = e.i32(node.in(1));
            IntArray ends = e.i32(node.in(2));
            FloatArray y = e.f32(node.out(0));
            int mode = node.iattr[0];
            IntArray arg = mode == IndexKernels.MAX ? e.i32(node.out(1)) : e.constant(0);
            int dim = node.out(0).dim(1);
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, IndexKernels::segmentReduce, e.context(), x, starts, ends, y, arg, dim, mode, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            IntArray starts = (IntArray) node.in(1).storage().host();
            IntArray ends = (IntArray) node.in(2).storage().host();
            FloatArray y = (FloatArray) node.out(0).storage().host();
            int mode = node.iattr[0];
            IntArray arg = mode == IndexKernels.MAX ? (IntArray) node.out(1).storage().host() : null;
            int dim = node.out(0).dim(1);
            for (int i = 0; i < node.out(0).numel(); i++) {
                int b = i / dim, d = i % dim;
                double acc = 0;
                double best = Double.NEGATIVE_INFINITY;
                int at = -1;
                for (int r = starts.get(b); r < ends.get(b); r++) {
                    double v = x.get(r * dim + d);
                    acc += v;
                    if (v > best) {
                        best = v;
                        at = r;
                    }
                }
                int count = ends.get(b) - starts.get(b);
                if (arg != null) {
                    y.set(i, at < 0 ? 0 : (float) best);
                    arg.set(i, at);
                } else {
                    y.set(i, (float) (mode == IndexKernels.MEAN ? (count > 0 ? acc / count : 0) : acc));
                }
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor gy) {
            Tensor dx = Tensor.result(node.in(0).shape(), DType.F32);
            Tensor arg = node.iattr[0] == IndexKernels.MAX ? node.out(1) : node.in(1);
            Tensor.record(SEGMENT_BACKWARD, new Tensor[] { gy, node.in(3), node.in(1), node.in(2), arg }, new Tensor[] { dx }, node.iattr.clone(), null, true);
            return new Tensor[] { dx, null, null, null };
        }
    }

    /** inputs [dy, rowSeg, starts, ends, argmax (or a placeholder)]; iattr = [mode]. */
    static final class SegmentReduceBackward implements Op {

        @Override
        public String name() {
            return "segment_reduce_bwd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray dy = e.f32(node.in(0));
            IntArray rowSeg = e.i32(node.in(1));
            IntArray starts = e.i32(node.in(2));
            IntArray ends = e.i32(node.in(3));
            IntArray arg = e.i32(node.in(4));
            FloatArray dx = e.f32(node.out(0));
            int mode = node.iattr[0];
            int dim = node.out(0).dim(1);
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, IndexKernels::segmentReduceBackward, e.context(), dy, rowSeg, starts, ends, arg, dx, dim, mode, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray dy = (FloatArray) node.in(0).storage().host();
            IntArray rowSeg = (IntArray) node.in(1).storage().host();
            IntArray starts = (IntArray) node.in(2).storage().host();
            IntArray ends = (IntArray) node.in(3).storage().host();
            IntArray arg = (IntArray) node.in(4).storage().host();
            FloatArray dx = (FloatArray) node.out(0).storage().host();
            int mode = node.iattr[0];
            int dim = node.out(0).dim(1);
            for (int i = 0; i < node.out(0).numel(); i++) {
                int r = i / dim, d = i % dim;
                int b = rowSeg.get(r);
                double v = 0;
                if (b >= 0) {
                    double g = dy.get(b * dim + d);
                    v = mode == IndexKernels.MAX ? (arg.get(b * dim + d) == r ? g : 0) : (mode == IndexKernels.MEAN ? g / (ends.get(b) - starts.get(b)) : g);
                }
                dx.set(i, (float) v);
            }
        }
    }
}
