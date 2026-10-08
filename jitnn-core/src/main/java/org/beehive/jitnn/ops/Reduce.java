package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.ReduceKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Sum, mean, max and min along one axis, and argmax. */
public final class Reduce implements Op {

    public enum Kind {
        SUM,
        MEAN,
        MAX,
        MIN
    }

    static final Reduce OP = new Reduce();
    static final ArgMax ARGMAX = new ArgMax();

    // iattr = [kernel op, outer, len, inner, isMean]
    private Reduce() {
    }

    public static Tensor apply(Kind kind, Tensor x, int axis, boolean keepDim) {
        Checks.f32(x);
        Shape s = x.shape();
        if (s.rank() == 0) {
            x = x.reshape(1);
            s = x.shape();
        }
        int a = s.axis(axis);
        int[] geo = geometry(s, a);
        int kop = switch (kind) {
            case SUM, MEAN -> ReduceKernels.SUM;
            case MAX -> ReduceKernels.MAX;
            case MIN -> ReduceKernels.MIN;
        };
        float scale = kind == Kind.MEAN ? 1.0f / geo[1] : 1.0f;
        Tensor out = Tensor.result(s.reduce(a, keepDim), DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, new int[] { kop, geo[0], geo[1], geo[2], kind == Kind.MEAN ? 1 : 0, a }, new float[] { scale }, true);
        return out;
    }

    public static Tensor argmax(Tensor x, int axis) {
        Checks.f32(x);
        int a = x.shape().axis(axis);
        int[] geo = geometry(x.shape(), a);
        Tensor out = Tensor.result(x.shape().reduce(a, false), DType.I32);
        Tensor.record(ARGMAX, new Tensor[] { x }, new Tensor[] { out }, geo, null, true);
        return out;
    }

    /** [outer, len, inner] for reducing {@code axis}. */
    static int[] geometry(Shape s, int axis) {
        int outer = 1;
        int inner = 1;
        int[] d = s.dims();
        for (int i = 0; i < axis; i++) {
            outer *= d[i];
        }
        for (int i = axis + 1; i < d.length; i++) {
            inner *= d[i];
        }
        return new int[] { outer, d[axis], inner };
    }

    @Override
    public String name() {
        return "reduce";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray out = e.f32(n.out(0));
        int op = n.iattr[0];
        int outer = n.iattr[1];
        int len = n.iattr[2];
        int inner = n.iattr[3];
        float scale = n.fattr[0];
        int outputs = outer * inner;
        if (inner == 1) {
            // Contiguous rows: a work-group per row, or per row chunk when rows are few and long.
            int parts = splitParts(outputs * Emitter.LOCAL, len, Emitter.LOCAL * 8, 1024);
            if (parts > 1) {
                int chunk = Emitter.ceilDiv(len, parts);
                FloatArray partial = e.scratch(outputs * parts);
                e.task(Emitter.groups(outputs * parts), (g, id) -> g.task(id, ReduceKernels::reduceRowsSplit, e.context(), x, partial, len, parts, chunk, op));
                e.task(Emitter.groups(outputs), (g, id) -> g.task(id, ReduceKernels::reduceGroup, e.context(), partial, out, outputs, parts, 1, op, scale));
            } else if (len >= 64) {
                e.task(Emitter.groups(outputs), (g, id) -> g.task(id, ReduceKernels::reduceGroup, e.context(), x, out, outer, len, inner, op, scale));
            } else {
                e.task(Emitter.grid1D(outputs), (g, id) -> g.task(id, ReduceKernels::reduceThread, e.context(), x, out, outer, len, inner, op, scale));
            }
        } else {
            if (COLUMN_GROUPS && len >= 64) {
                // Strided axis by work-groups of 32 columns x 8 row lanes, split along the axis
                // so that about 4 groups per SM have work; then the parts summed in order.
                int tiles = Emitter.ceilDiv(outputs, ReduceKernels.COL_TILE);
                int parts = Math.max(1, Math.min(Emitter.ceilDiv(432, tiles), len / 64));
                int chunk = Emitter.ceilDiv(len, parts);
                int sl = Emitter.ceilDiv(len, chunk);
                FloatArray partial = sl > 1 ? e.scratch(outputs * sl) : null;
                if (sl > 1) {
                    e.task(Emitter.blocks1D(sl * tiles, 256), (g, id) -> g.task(id, ReduceKernels::reduceColumnsGroup, e.context(), x, partial, outer, len, inner, sl, chunk, op));
                    e.task(Emitter.grid1D(outputs), (g, id) -> g.task(id, ReduceKernels::reduceThread, e.context(), partial, out, 1, sl, outputs, op, scale));
                    return;
                }
            }
            // Strided axis: a thread per output column (coalesced), split along the axis when
            // there are too few columns to fill the device.
            int parts = splitParts(outputs, len, 64, 256);
            if (parts > 1) {
                int chunk = Emitter.ceilDiv(len, parts);
                FloatArray partial = e.scratch(outputs * parts);
                e.task(Emitter.grid1D(outputs * parts), (g, id) -> g.task(id, ReduceKernels::reduceColumnsSplit, e.context(), x, partial, outer, len, inner, parts, chunk, op));
                e.task(Emitter.grid1D(outputs), (g, id) -> g.task(id, ReduceKernels::reduceThread, e.context(), partial, out, 1, parts, outputs, op, scale));
            } else {
                e.task(Emitter.grid1D(outputs), (g, id) -> g.task(id, ReduceKernels::reduceThread, e.context(), x, out, outer, len, inner, op, scale));
            }
        }
    }

    /** Work-group column reductions ({@link ReduceKernels#reduceColumnsGroup}) unless {@code -Djitnn.reduceCols=thread}. */
    static final boolean COLUMN_GROUPS = !"thread".equals(System.getProperty("jitnn.reduceCols"));

    /**
     * How many chunks to split each output's axis into so that about 2^16 threads have work:
     * {@code threads} is the thread count without splitting, each chunk keeps at least
     * {@code minChunk} elements, and there are at most {@code maxParts} chunks. 1 means no split.
     */
    static int splitParts(int threads, int len, int minChunk, int maxParts) {
        final int target = 1 << 16;
        if (threads >= target) {
            return 1;
        }
        int parts = Math.min(Math.min(Emitter.ceilDiv(target, threads), len / minChunk), maxParts);
        return Math.max(parts, 1);
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray o = (FloatArray) n.out(0).storage().host();
        int op = n.iattr[0];
        int outer = n.iattr[1];
        int len = n.iattr[2];
        int inner = n.iattr[3];
        for (int oi = 0; oi < outer; oi++) {
            for (int j = 0; j < inner; j++) {
                double acc = op == ReduceKernels.MAX ? Double.NEGATIVE_INFINITY : op == ReduceKernels.MIN ? Double.POSITIVE_INFINITY : 0;
                for (int k = 0; k < len; k++) {
                    double v = x.get((oi * len + k) * inner + j);
                    acc = op == ReduceKernels.MAX ? Math.max(acc, v) : op == ReduceKernels.MIN ? Math.min(acc, v) : acc + v;
                }
                o.set(oi * inner + j, (float) (op == ReduceKernels.SUM ? acc * n.fattr[0] : acc));
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor x = n.in(0);
        int axis = n.iattr[5];
        Shape keep = x.shape().reduce(axis, true);
        Tensor g = gy.reshape(keep.dims()).expand(x.shape());
        if (n.iattr[0] == ReduceKernels.SUM) {
            return new Tensor[] { n.iattr[4] == 1 ? g.mul(n.fattr[0]) : g };
        }
        // max/min: the gradient goes to every element equal to the extremum.
        Tensor extremum = n.out(0).reshape(keep.dims()).expand(x.shape());
        return new Tensor[] { g.mul(x.eq(extremum)) };
    }

    /** Argmax along an axis; iattr = [outer, len, inner]. */
    static final class ArgMax implements Op {

        @Override
        public String name() {
            return "argmax";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            IntArray out = e.i32(n.out(0));
            int outer = n.iattr[0];
            int len = n.iattr[1];
            int inner = n.iattr[2];
            e.task(Emitter.grid1D(outer * inner), (g, id) -> g.task(id, ReduceKernels::argmaxThread, e.context(), x, out, outer, len, inner));
        }

        @Override
        public void reference(Node n) {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            IntArray o = (IntArray) n.out(0).storage().host();
            int outer = n.iattr[0];
            int len = n.iattr[1];
            int inner = n.iattr[2];
            for (int oi = 0; oi < outer; oi++) {
                for (int j = 0; j < inner; j++) {
                    int best = 0;
                    for (int k = 1; k < len; k++) {
                        if (x.get((oi * len + k) * inner + j) > x.get((oi * len + best) * inner + j)) {
                            best = k;
                        }
                    }
                    o.set(oi * inner + j, best);
                }
            }
        }
    }
}
