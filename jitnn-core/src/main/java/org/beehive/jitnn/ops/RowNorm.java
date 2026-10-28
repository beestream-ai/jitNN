package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.NormKernels;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * LayerNorm and RMSNorm over the last axis, PyTorch semantics (biased variance; RMSNorm divides
 * by sqrt(mean(x^2) + eps)). Inputs [x, weight] or [x, weight, bias]; outputs [y, mean, rstd],
 * one statistic per row. iattr = [rows, cols, mode, hasBias]; fattr = [eps].
 */
public final class RowNorm implements Op {

    static final RowNorm OP = new RowNorm();
    static final Backward BACKWARD = new Backward();

    private RowNorm() {
    }

    public static Tensor layerNorm(Tensor x, Tensor weight, Tensor bias, float eps) {
        return apply(x, weight, bias, eps, NormKernels.LAYER);
    }

    public static Tensor rmsNorm(Tensor x, Tensor weight, float eps) {
        return apply(x, weight, null, eps, NormKernels.RMS);
    }

    private static Tensor apply(Tensor x, Tensor weight, Tensor bias, float eps, int mode) {
        Checks.f32(x, weight);
        int cols = x.dim(-1);
        int rows = x.numel() / Math.max(cols, 1);
        if (weight.numel() != cols || (bias != null && bias.numel() != cols)) {
            throw new IllegalArgumentException("norm over " + cols + " features with weight " + weight.shape() + (bias != null ? " and bias " + bias.shape() : ""));
        }
        // half-precision inference: FP16 out (from an FP16 or FP32 input); statistics stay FP32
        boolean half = org.beehive.jitnn.Autocast.halfActivations();
        Tensor y = Tensor.result(x.shape(), half ? DType.F16 : DType.F32);
        Tensor mean = Tensor.result(Shape.of(rows), DType.F32);
        Tensor rstd = Tensor.result(Shape.of(rows), DType.F32);
        Tensor[] in = bias != null ? new Tensor[] { x, weight, bias } : new Tensor[] { x, weight };
        Tensor.record(OP, in, new Tensor[] { y, mean, rstd }, new int[] { rows, cols, mode, bias != null ? 1 : 0 }, new float[] { eps }, true);
        return y;
    }

    @Override
    public String name() {
        return "row_norm";
    }

    @Override
    public boolean acceptsHalf() {
        return true;
    }

    @Override
    public void emit(Emitter e, Node n) {
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        int mode = n.iattr[2];
        int hasBias = n.iattr[3];
        float eps = n.fattr[0];
        FloatArray w = e.f32(n.in(1));
        FloatArray b = hasBias != 0 ? e.f32(n.in(2)) : e.dummyF32();
        if (n.iattr.length > 4 && n.iattr[4] >= ADD_BIAS) {
            // x + residual + the producing projection's bias, then the norm: inputs [x, w, (b), residual, projection bias]
            FloatArray mh = e.f32(n.out(1));
            FloatArray rh = e.f32(n.out(2));
            Tensor lbT = n.in(n.inputs.length - 1);
            FloatArray lb = lbT.dtype() == DType.F16 ? halfToFloat(e, e.f16(lbT), cols) : e.f32(lbT);
            if (n.out(0).dtype() == DType.F16) {
                var xh = e.f16(n.in(0));
                var res = e.f16(n.in(n.inputs.length - 2));
                var yh = e.f16(n.out(0));
                e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormForwardHalfAddBias, e.context(), xh, res, lb, w, b, yh, mh, rh, cols, eps, mode, hasBias));
            } else {
                FloatArray xf = e.f32(n.in(0));
                FloatArray res = e.f32(n.in(n.inputs.length - 2));
                FloatArray yf = e.f32(n.out(0));
                e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormForwardAddBias, e.context(), xf, res, lb, w, b, yf, mh, rh, cols, eps, mode, hasBias));
            }
            return;
        }
        if (n.out(0).dtype() == DType.F16) {
            var yh = e.f16(n.out(0));
            FloatArray mh = e.f32(n.out(1));
            FloatArray rh = e.f32(n.out(2));
            // a warp per row on CUDA with many rows (FP16 rows are short to read), a work-group per row otherwise
            boolean warp = WARP && e.isCuda() && rows >= 256 && cols <= 4096;
            if (WARP && e.isCuda()) {
                warp = org.beehive.jitnn.engine.Autotune.choose(e, "rownormHalf", rows + "x" + cols, warp ? "warp" : "group", "warp", "group").equals("warp");
            }
            WorkerGrid wg = Emitter.blocks1D(Emitter.ceilDiv(rows, NormKernels.WARP_ROWS), 32 * NormKernels.WARP_ROWS);
            if (n.iattr.length > 4 && n.iattr[4] != 0) {
                // x + residual (an add fused into the norm by Fusion), both FP16
                var xh = e.f16(n.in(0));
                var res = e.f16(n.in(n.inputs.length - 1));
                if (warp) {
                    e.task(wg, (g, id) -> g.task(id, NormKernels::rowNormForwardHalfAddWarp, e.context(), xh, res, w, b, yh, mh, rh, rows, cols, eps, mode, hasBias));
                } else {
                    e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormForwardHalfAdd, e.context(), xh, res, w, b, yh, mh, rh, cols, eps, mode, hasBias));
                }
            } else if (n.in(0).dtype() == DType.F16) {
                var xh = e.f16(n.in(0));
                if (warp) {
                    e.task(wg, (g, id) -> g.task(id, NormKernels::rowNormForwardHalfWarp, e.context(), xh, w, b, yh, mh, rh, rows, cols, eps, mode, hasBias));
                } else {
                    e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormForwardHalf, e.context(), xh, w, b, yh, mh, rh, cols, eps, mode, hasBias));
                }
            } else {
                FloatArray xf = e.f32(n.in(0));
                if (warp) {
                    e.task(wg, (g, id) -> g.task(id, NormKernels::rowNormForwardToHalfWarp, e.context(), xf, w, b, yh, mh, rh, rows, cols, eps, mode, hasBias));
                } else {
                    e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormForwardToHalf, e.context(), xf, w, b, yh, mh, rh, cols, eps, mode, hasBias));
                }
            }
            return;
        }
        FloatArray x = e.f32(n.in(0));
        FloatArray y = e.f32(n.out(0));
        if (e.unread(n.out(1)) && e.unread(n.out(2))
                && org.beehive.jitnn.natives.Libs.rowNorm(e, x, w, hasBias != 0 ? b : null, y, rows, cols, eps, mode == NormKernels.LAYER)) {
            return;
        }
        FloatArray mean = e.f32(n.out(1));
        FloatArray rstd = e.f32(n.out(2));
        // the warp forward wins with many rows of moderate width (A100: 2048-4096 x 768); with few
        // rows (128) or wide ones (1024 x 1024) a work-group per row is faster
        boolean warp = WARP && e.isCuda() && rows >= 1024 && cols <= 768;
        if (WARP && e.isCuda()) {
            warp = org.beehive.jitnn.engine.Autotune.choose(e, "rownorm", rows + "x" + cols, warp ? "warp" : "group", "warp", "group").equals("warp");
        }
        if (warp) {
            WorkerGrid grid = Emitter.blocks1D(Emitter.ceilDiv(rows, NormKernels.WARP_ROWS), 32 * NormKernels.WARP_ROWS);
            e.task(grid, (g, id) -> g.task(id, NormKernels::rowNormForwardWarp, e.context(), x, w, b, y, mean, rstd, rows, cols, eps, mode, hasBias));
            return;
        }
        e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormForward, e.context(), x, w, b, y, mean, rstd, cols, eps, mode, hasBias));
    }

    /** iattr[4] of a norm of x + residual + the bias of the projection that produced one of them (inputs [x, w, (b), residual, bias]). */
    static final int ADD_BIAS = 2;

    private static FloatArray halfToFloat(Emitter e, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray h, int count) {
        FloatArray f = e.scratch(count);
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f16ToF32, e.context(), h, f, count));
        return f;
    }

    /**
     * On Metal with MLX's products (whose bias is a pass of its own): a linear layer's bias moved into the
     * norm that reads it through a residual add. FP32: linear, add and norm become the norm of x + residual
     * + bias (the add goes away); FP16, where the add is already in the norm: the bias joins it. The linear
     * output and the add's must have no other reader, and the norm's statistics none.
     */
    public static java.util.List<Node> foldResidualBias(java.util.List<Node> nodes, java.util.Set<org.beehive.jitnn.engine.Storage> wanted) {
        java.util.IdentityHashMap<org.beehive.jitnn.engine.Storage, Integer> uses = new java.util.IdentityHashMap<>();
        java.util.IdentityHashMap<org.beehive.jitnn.engine.Storage, Integer> reader = new java.util.IdentityHashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (Tensor t : nodes.get(i).inputs) {
                uses.merge(t.storage(), 1, Integer::sum);
                reader.put(t.storage(), i);
            }
        }
        java.util.List<Node> out = new java.util.ArrayList<>(nodes);
        boolean changed = false;
        for (int i = 0; i < out.size(); i++) {
            Node l = out.get(i);
            if (l == null || l.op != Linear.OP || l.iattr[3] == 0 || l.iattr[4] != Linear.NONE || wanted.contains(l.out(0).storage())) {
                continue;
            }
            boolean half = l.out(0).dtype() == DType.F16;
            Integer j = reader.get(l.out(0).storage());
            if (j == null || j <= i || out.get(j) == null || uses.getOrDefault(l.out(0).storage(), 0) != 1
                    || !org.beehive.jitnn.engine.Native.use(half ? "gemmHalf" : "gemm", org.beehive.jitnn.engine.Native.Lib.MLX)) {
                continue;
            }
            Node r = out.get(j);
            Tensor y = l.out(0);
            Node norm;
            Tensor x;
            Tensor res;
            int normAt;
            if (!half && r.op == Binary.OP && r.inputs.length == 2 && r.iattr[0] == Binary.Kind.ADD.code && r.in(0).shape().equals(r.in(1).shape())
                    && r.out(0).dtype() == DType.F32 && r.in(0).dtype() == DType.F32 && r.in(1).dtype() == DType.F32 && !wanted.contains(r.out(0).storage())) {
                Integer k = reader.get(r.out(0).storage());
                if (k == null || k <= j || out.get(k) == null || uses.getOrDefault(r.out(0).storage(), 0) != 1) {
                    continue;
                }
                norm = out.get(k);
                if (norm.op != OP || norm.iattr.length != 4 || norm.in(0).storage() != r.out(0).storage() || norm.out(0).dtype() != DType.F32) {
                    continue;
                }
                x = y;
                res = r.in(0).storage() == y.storage() ? r.in(1) : r.in(0);
                normAt = k;
            } else if (half && r.op == OP && r.iattr.length == 5 && r.iattr[4] == 1 && r.out(0).dtype() == DType.F16) {
                norm = r;
                Tensor a = r.in(0);
                Tensor last = r.in(r.inputs.length - 1);
                if (a.storage() != y.storage() && last.storage() != y.storage()) {
                    continue;
                }
                x = y;
                res = a.storage() == y.storage() ? last : a;
                normAt = j;
            } else {
                continue;
            }
            if (uses.containsKey(norm.out(1).storage()) || uses.containsKey(norm.out(2).storage()) || wanted.contains(norm.out(1).storage())
                    || wanted.contains(norm.out(2).storage())) {
                continue;
            }
            int[] la = l.iattr.clone();
            la[3] = 0;
            out.set(i, new Node(Linear.OP, new Tensor[] { l.in(0), l.in(1) }, l.outputs, la, l.fattr, true));
            int hasNormBias = norm.iattr[3];
            Tensor[] in = hasNormBias != 0 ? new Tensor[] { x, norm.in(1), norm.in(2), res, l.in(2) } : new Tensor[] { x, norm.in(1), res, l.in(2) };
            int[] na = java.util.Arrays.copyOf(norm.iattr, 5);
            na[4] = ADD_BIAS;
            out.set(normAt, new Node(OP, in, norm.outputs, na, norm.fattr, true));
            if (normAt != j) {
                out.set(j, null);
            }
            changed = true;
        }
        if (!changed) {
            return nodes;
        }
        java.util.List<Node> kept = new java.util.ArrayList<>(out.size());
        for (Node n : out) {
            if (n != null) {
                kept.add(n);
            }
        }
        return kept;
    }

    /** Warp-per-row kernels with warp reductions on CUDA unless {@code -Djitnn.rownorm=group}. */
    static final boolean WARP = !"group".equals(System.getProperty("jitnn.rownorm"));

    @Override
    public void reference(Node n) {
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        boolean layer = n.iattr[2] == NormKernels.LAYER;
        boolean hasBias = n.iattr[3] != 0;
        double eps = n.fattr[0];
        Object xIn = n.in(0).storage().host();
        Object res = n.iattr.length > 4 && n.iattr[4] != 0 ? n.in(n.inputs.length - 1).storage().host() : null;
        int len = rows * cols;
        float[] xs = new float[len];
        for (int i = 0; i < len; i++) {
            xs[i] = HostValues.get(xIn, i) + (res != null ? HostValues.get(res, i) : 0.0f);
        }
        FloatArray x = FloatArray.fromArray(xs);
        FloatArray w = (FloatArray) n.in(1).storage().host();
        FloatArray b = hasBias ? (FloatArray) n.in(2).storage().host() : null;
        Object y = n.out(0).storage().host();
        FloatArray mean = (FloatArray) n.out(1).storage().host();
        FloatArray rstd = (FloatArray) n.out(2).storage().host();
        for (int r = 0; r < rows; r++) {
            double mu = 0;
            if (layer) {
                for (int c = 0; c < cols; c++) {
                    mu += x.get(r * cols + c);
                }
                mu /= cols;
            }
            double sq = 0;
            for (int c = 0; c < cols; c++) {
                double d = x.get(r * cols + c) - mu;
                sq += d * d;
            }
            double is = 1 / Math.sqrt(sq / cols + eps);
            for (int c = 0; c < cols; c++) {
                double v = (x.get(r * cols + c) - mu) * is * w.get(c) + (hasBias ? b.get(c) : 0);
                HostValues.set(y, r * cols + c, (float) v);
            }
            mean.set(r, (float) mu);
            rstd.set(r, (float) is);
        }
    }

    @Override
    public boolean cpu(Node n) {
        boolean residual = n.iattr.length > 4 && n.iattr[4] != 0;
        return CpuKernels.rowNorm(n.in(0), residual ? n.in(n.inputs.length - 1) : null, n.in(1), n.iattr[3] != 0 ? n.in(2) : null, n.out(0), n.out(1), n.out(2), n.iattr[0], n.iattr[1],
                n.iattr[2] == NormKernels.LAYER, n.fattr[0]);
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        boolean hasBias = n.iattr[3] != 0;
        Tensor x = n.in(0);
        Tensor dx = Tensor.result(x.shape(), DType.F32);
        Tensor dw = Tensor.result(n.in(1).shape(), DType.F32);
        Tensor[] outs = hasBias ? new Tensor[] { dx, dw, Tensor.result(n.in(2).shape(), DType.F32) } : new Tensor[] { dx, dw };
        // FP16 activations: x is read as FP16; the gradient and the weight are FP32
        Tensor dy = gy.dtype() == DType.F32 ? gy : gy.to(DType.F32);
        Tensor w = n.in(1).dtype() == DType.F32 ? n.in(1) : n.in(1).to(DType.F32);
        boolean device = org.beehive.jitnn.engine.Engine.backend() == org.beehive.jitnn.engine.Engine.Backend.TORNADO;
        Tensor xin = x.dtype() == DType.F32 || (x.dtype() == DType.F16 && device) ? x : x.to(DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { dy, xin, w, n.out(1), n.out(2) }, outs, n.iattr.clone(), null, true);
        Tensor[] grads = new Tensor[n.inputs.length];
        grads[0] = x.requiresGrad() ? dx : null;
        grads[1] = n.in(1).requiresGrad() ? dw : null;
        if (hasBias) {
            grads[2] = n.in(2).requiresGrad() ? outs[2] : null;
        }
        return grads;
    }

    /** Inputs [dy, x, weight, mean, rstd]; outputs [dx, dweight] or [dx, dweight, dbias]. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "row_norm_bwd";
        }

        /** x may be FP16 (FP16 activations); everything else is FP32. */
        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node n) {
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            int mode = n.iattr[2];
            boolean hasBias = n.iattr[3] != 0;
            FloatArray dy = e.f32(n.in(0));
            boolean hx = n.in(1).dtype() == DType.F16;
            FloatArray x = hx ? null : e.f32(n.in(1));
            var xh = hx ? e.f16(n.in(1)) : null;
            FloatArray w = e.f32(n.in(2));
            FloatArray mean = e.f32(n.in(3));
            FloatArray rstd = e.f32(n.in(4));
            FloatArray dx = e.f32(n.out(0));
            FloatArray dw = e.f32(n.out(1));
            if (WARP && e.isCuda() && rows >= 1024 && cols <= 768) {
                // dx: a warp per row (where it wins, as in the forward); dw, db: 16-row partials,
                // then both sums in one launch
                int chunks = Emitter.ceilDiv(rows, NormKernels.PARAM_ROWS);
                FloatArray partW = e.scratch(chunks * cols);
                FloatArray partB = e.scratch(chunks * cols);
                int parts = chunks * cols;
                FloatArray db = hasBias ? e.f32(n.out(2)) : e.dummyF32();
                int bias = hasBias ? 1 : 0;
                WorkerGrid grid = Emitter.blocks1D(Emitter.ceilDiv(rows, NormKernels.WARP_ROWS), 32 * NormKernels.WARP_ROWS);
                if (hx) {
                e.task(grid, (g, id) -> g.task(id, NormKernels::rowNormBackwardWarpHalfX, e.context(), dy, xh, w, mean, rstd, dx, rows, cols, mode));
            } else {
                e.task(grid, (g, id) -> g.task(id, NormKernels::rowNormBackwardWarp, e.context(), dy, x, w, mean, rstd, dx, rows, cols, mode));
            }
                if (hx) {
                e.task(Emitter.grid1D(parts), (g, id) -> g.task(id, NormKernels::rowNormParamPartialsChunkedHalfX, e.context(), dy, xh, mean, rstd, partW, partB, rows, cols, parts));
            } else {
                e.task(Emitter.grid1D(parts), (g, id) -> g.task(id, NormKernels::rowNormParamPartialsChunked, e.context(), dy, x, mean, rstd, partW, partB, rows, cols, parts));
            }
                e.task(Emitter.grid1D(2 * cols), (g, id) -> g.task(id, NormKernels::sumPartialsPair, e.context(), partW, partB, dw, db, chunks, cols, bias));
                return;
            }
            int chunks = Emitter.ceilDiv(rows, NormKernels.PARAM_CHUNK);
            FloatArray partW = e.scratch(chunks * cols);
            FloatArray partB = e.scratch(chunks * cols);
            int parts = chunks * cols;
            if (hx) {
                e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormBackwardHalfX, e.context(), dy, xh, w, mean, rstd, dx, cols, mode));
            } else {
                e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormBackward, e.context(), dy, x, w, mean, rstd, dx, cols, mode));
            }
            if (hx) {
                e.task(Emitter.grid1D(parts), (g, id) -> g.task(id, NormKernels::rowNormParamPartialsHalfX, e.context(), dy, xh, mean, rstd, partW, partB, rows, cols, parts));
            } else {
                e.task(Emitter.grid1D(parts), (g, id) -> g.task(id, NormKernels::rowNormParamPartials, e.context(), dy, x, mean, rstd, partW, partB, rows, cols, parts));
            }
            e.task(Emitter.grid1D(cols), (g, id) -> g.task(id, NormKernels::sumPartials, e.context(), partW, dw, chunks, cols));
            if (hasBias) {
                FloatArray db = e.f32(n.out(2));
                e.task(Emitter.grid1D(cols), (g, id) -> g.task(id, NormKernels::sumPartials, e.context(), partB, db, chunks, cols));
            }
        }

        @Override
        public void reference(Node n) {
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            boolean layer = n.iattr[2] == NormKernels.LAYER;
            boolean hasBias = n.iattr[3] != 0;
            FloatArray dy = (FloatArray) n.in(0).storage().host();
            FloatArray x = (FloatArray) n.in(1).storage().host();
            FloatArray w = (FloatArray) n.in(2).storage().host();
            FloatArray mean = (FloatArray) n.in(3).storage().host();
            FloatArray rstd = (FloatArray) n.in(4).storage().host();
            FloatArray dx = (FloatArray) n.out(0).storage().host();
            double[] dw = new double[cols];
            double[] db = new double[cols];
            for (int r = 0; r < rows; r++) {
                double mu = mean.get(r);
                double is = rstd.get(r);
                double sa = 0;
                double sb = 0;
                for (int c = 0; c < cols; c++) {
                    double g = (double) dy.get(r * cols + c) * w.get(c);
                    double xhat = (x.get(r * cols + c) - mu) * is;
                    sa += g;
                    sb += g * xhat;
                    dw[c] += dy.get(r * cols + c) * xhat;
                    db[c] += dy.get(r * cols + c);
                }
                double meanG = layer ? sa / cols : 0;
                double meanGX = sb / cols;
                for (int c = 0; c < cols; c++) {
                    double g = (double) dy.get(r * cols + c) * w.get(c);
                    double xhat = (x.get(r * cols + c) - mu) * is;
                    dx.set(r * cols + c, (float) (is * (g - meanG - xhat * meanGX)));
                }
            }
            FloatArray dwOut = (FloatArray) n.out(1).storage().host();
            for (int c = 0; c < cols; c++) {
                dwOut.set(c, (float) dw[c]);
            }
            if (hasBias) {
                FloatArray dbOut = (FloatArray) n.out(2).storage().host();
                for (int c = 0; c < cols; c++) {
                    dbOut.set(c, (float) db[c]);
                }
            }
        }
    }
}
