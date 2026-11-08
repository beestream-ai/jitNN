package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.NdIndexKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GatherND and ScatterND (ONNX, batch_dims 0). For data [d0, ..., dn-1] and INT32 indices [..., k]
 * (k <= n), each index row picks the slice data[i0, ..., ik-1, :, ...]: GatherND returns them as
 * [..., d_k, ..., dn-1]; ScatterND writes, adds or multiplies the update slices into a copy of
 * data. iattr = [k, rows, slice, reduction (ScatterND)].
 */
public final class NdIndexing {

    static final GatherND GATHER = new GatherND();
    static final ScatterND SCATTER = new ScatterND();

    private NdIndexing() {
    }

    /** [k, rows, slice] after checking the shapes. */
    private static int[] geometry(String op, Tensor data, Tensor indices) {
        Checks.f32(data);
        if (indices.dtype() != DType.I32 || indices.rank() < 1) {
            throw new IllegalArgumentException(op + ": indices must be INT32 [..., k], got " + indices);
        }
        int k = indices.dim(-1);
        if (k < 1 || k > data.rank()) {
            throw new IllegalArgumentException(op + ": index rows of " + k + " for data " + data.shape());
        }
        int slice = 1;
        for (int a = k; a < data.rank(); a++) {
            slice *= data.dim(a);
        }
        return new int[] { k, indices.numel() / k, slice };
    }

    /** meta for the offset kernel: [k, dims[0..k), strides[0..k)]. */
    static int[] offsetMeta(Shape data, int k) {
        int[] d = data.dims();
        int[] meta = new int[1 + 2 * k];
        meta[0] = k;
        int stride = 1;
        for (int a = d.length - 1; a >= 0; a--) {
            if (a < k) {
                meta[1 + a] = d[a];
                meta[1 + k + a] = stride;
            }
            stride *= d[a];
        }
        return meta;
    }

    static int[] hostOffsets(Tensor data, IntArray idx, int[] g) {
        int[] meta = offsetMeta(data.shape(), g[0]);
        int k = g[0];
        int[] off = new int[g[1]];
        for (int r = 0; r < g[1]; r++) {
            for (int t = 0; t < k; t++) {
                int v = idx.get(r * k + t);
                off[r] += (v < 0 ? v + meta[1 + t] : v) * meta[1 + k + t];
            }
        }
        return off;
    }

    public static Tensor gatherND(Tensor data, Tensor indices) {
        int[] g = geometry("gatherND", data, indices);
        int[] id = indices.shape().dims();
        int[] dd = data.shape().dims();
        int[] dims = new int[id.length - 1 + dd.length - g[0]];
        System.arraycopy(id, 0, dims, 0, id.length - 1);
        System.arraycopy(dd, g[0], dims, id.length - 1, dd.length - g[0]);
        Tensor out = Tensor.result(Shape.of(dims), DType.F32);
        Tensor.record(GATHER, new Tensor[] { data, indices }, new Tensor[] { out }, new int[] { g[0], g[1], g[2] }, null, true);
        return out;
    }

    /** reduction: "none", "add" or "mul". */
    public static Tensor scatterND(Tensor data, Tensor indices, Tensor updates, String reduction) {
        int[] g = geometry("scatterND", data, indices);
        Checks.f32(updates);
        if (updates.numel() != g[1] * g[2]) {
            throw new IllegalArgumentException("scatterND: updates " + updates.shape() + " do not match " + g[1] + " slices of " + g[2]);
        }
        int red = switch (reduction) {
            case "none" -> NdIndexKernels.NONE;
            case "add" -> NdIndexKernels.ADD;
            case "mul" -> NdIndexKernels.MUL;
            default -> throw new IllegalArgumentException("scatterND: reduction must be none, add or mul, got " + reduction);
        };
        Tensor out = Tensor.result(data.shape(), DType.F32);
        Tensor.record(SCATTER, new Tensor[] { data, indices, updates }, new Tensor[] { out }, new int[] { g[0], g[1], g[2], red }, null, true);
        return out;
    }

    static IntArray emitOffsets(Emitter e, Node n) {
        IntArray idx = e.i32(n.in(1));
        IntArray meta = e.constant(offsetMeta(n.in(0).shape(), n.iattr[0]));
        int rows = n.iattr[1];
        IntArray off = e.scratchI32(rows);
        e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, NdIndexKernels::ndOffsets, e.context(), idx, meta, off, rows));
        return off;
    }

    static final class GatherND implements Op {

        @Override
        public String name() {
            return "gather_nd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray off = emitOffsets(e, n);
            FloatArray data = e.f32(n.in(0));
            FloatArray out = e.f32(n.out(0));
            int slice = n.iattr[2];
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NdIndexKernels::gatherND, e.context(), data, off, out, slice, total));
        }

        @Override
        public void reference(Node n) {
            FloatArray data = (FloatArray) n.in(0).storage().host();
            FloatArray out = (FloatArray) n.out(0).storage().host();
            int[] off = hostOffsets(n.in(0), (IntArray) n.in(1).storage().host(), n.iattr);
            int slice = n.iattr[2];
            for (int i = 0; i < n.out(0).numel(); i++) {
                out.set(i, data.get(off[i / slice] + i % slice));
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor zeros = Tensor.zeros(n.in(0).shape().dims());
            return new Tensor[] { scatterND(zeros, n.in(1), gy, "add"), null };
        }
    }

    static final class ScatterND implements Op {

        @Override
        public String name() {
            return "scatter_nd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray off = emitOffsets(e, n);
            FloatArray data = e.f32(n.in(0));
            FloatArray upd = e.f32(n.in(2));
            FloatArray out = e.f32(n.out(0));
            int rows = n.iattr[1];
            int slice = n.iattr[2];
            int red = n.iattr[3];
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NdIndexKernels::scatterND, e.context(), data, off, upd, out, rows, slice, red, total));
        }

        @Override
        public void reference(Node n) {
            FloatArray data = (FloatArray) n.in(0).storage().host();
            FloatArray upd = (FloatArray) n.in(2).storage().host();
            FloatArray out = (FloatArray) n.out(0).storage().host();
            int[] off = hostOffsets(n.in(0), (IntArray) n.in(1).storage().host(), n.iattr);
            int slice = n.iattr[2];
            int red = n.iattr[3];
            float[] o = new float[n.out(0).numel()];
            for (int i = 0; i < o.length; i++) {
                o[i] = data.get(i);
            }
            for (int r = 0; r < off.length; r++) {
                for (int j = 0; j < slice; j++) {
                    float u = upd.get(r * slice + j);
                    int at = off[r] + j;
                    o[at] = red == NdIndexKernels.ADD ? o[at] + u : red == NdIndexKernels.MUL ? o[at] * u : u;
                }
            }
            for (int i = 0; i < o.length; i++) {
                out.set(i, o[i]);
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            int red = n.iattr[3];
            if (red == NdIndexKernels.MUL) {
                throw new UnsupportedOperationException("no gradient for scatterND with reduction mul");
            }
            Tensor data = n.in(0);
            Tensor indices = n.in(1);
            Tensor upd = n.in(2);
            Tensor gData = null;
            if (data.requiresGrad()) {
                // add: dy flows through unchanged; none: the overwritten positions get nothing
                gData = red == NdIndexKernels.ADD ? gy
                        : scatterND(gy, indices, Tensor.zeros(upd.shape().dims()), "none");
            }
            Tensor gUpd = upd.requiresGrad() ? gatherND(gy, indices).reshape(upd.shape().dims()) : null;
            return new Tensor[] { gData, null, gUpd };
        }
    }
}
