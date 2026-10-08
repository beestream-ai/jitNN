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
        Tensor y = Tensor.result(x.shape(), DType.F32);
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
    public void emit(Emitter e, Node n) {
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        int mode = n.iattr[2];
        int hasBias = n.iattr[3];
        float eps = n.fattr[0];
        FloatArray x = e.f32(n.in(0));
        FloatArray w = e.f32(n.in(1));
        FloatArray b = hasBias != 0 ? e.f32(n.in(2)) : e.dummyF32();
        FloatArray y = e.f32(n.out(0));
        FloatArray mean = e.f32(n.out(1));
        FloatArray rstd = e.f32(n.out(2));
        if (WARP && e.isCuda()) {
            WorkerGrid grid = Emitter.blocks1D(Emitter.ceilDiv(rows, NormKernels.WARP_ROWS), 32 * NormKernels.WARP_ROWS);
            e.task(grid, (g, id) -> g.task(id, NormKernels::rowNormForwardWarp, e.context(), x, w, b, y, mean, rstd, rows, cols, eps, mode, hasBias));
            return;
        }
        e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormForward, e.context(), x, w, b, y, mean, rstd, cols, eps, mode, hasBias));
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
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray w = (FloatArray) n.in(1).storage().host();
        FloatArray b = hasBias ? (FloatArray) n.in(2).storage().host() : null;
        FloatArray y = (FloatArray) n.out(0).storage().host();
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
                y.set(r * cols + c, (float) v);
            }
            mean.set(r, (float) mu);
            rstd.set(r, (float) is);
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        boolean hasBias = n.iattr[3] != 0;
        Tensor x = n.in(0);
        Tensor dx = Tensor.result(x.shape(), DType.F32);
        Tensor dw = Tensor.result(n.in(1).shape(), DType.F32);
        Tensor[] outs = hasBias ? new Tensor[] { dx, dw, Tensor.result(n.in(2).shape(), DType.F32) } : new Tensor[] { dx, dw };
        Tensor.record(BACKWARD, new Tensor[] { gy, x, n.in(1), n.out(1), n.out(2) }, outs, n.iattr.clone(), null, true);
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

        @Override
        public void emit(Emitter e, Node n) {
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            int mode = n.iattr[2];
            boolean hasBias = n.iattr[3] != 0;
            FloatArray dy = e.f32(n.in(0));
            FloatArray x = e.f32(n.in(1));
            FloatArray w = e.f32(n.in(2));
            FloatArray mean = e.f32(n.in(3));
            FloatArray rstd = e.f32(n.in(4));
            FloatArray dx = e.f32(n.out(0));
            FloatArray dw = e.f32(n.out(1));
            if (WARP && e.isCuda()) {
                // dx: a warp per row; dw, db: 16-row partials, then both sums in one launch
                int chunks = Emitter.ceilDiv(rows, NormKernels.PARAM_ROWS);
                FloatArray partW = e.scratch(chunks * cols);
                FloatArray partB = e.scratch(chunks * cols);
                int parts = chunks * cols;
                FloatArray db = hasBias ? e.f32(n.out(2)) : e.dummyF32();
                int bias = hasBias ? 1 : 0;
                WorkerGrid grid = Emitter.blocks1D(Emitter.ceilDiv(rows, NormKernels.WARP_ROWS), 32 * NormKernels.WARP_ROWS);
                e.task(grid, (g, id) -> g.task(id, NormKernels::rowNormBackwardWarp, e.context(), dy, x, w, mean, rstd, dx, rows, cols, mode));
                e.task(Emitter.grid1D(parts), (g, id) -> g.task(id, NormKernels::rowNormParamPartialsChunked, e.context(), dy, x, mean, rstd, partW, partB, rows, cols, parts));
                e.task(Emitter.grid1D(2 * cols), (g, id) -> g.task(id, NormKernels::sumPartialsPair, e.context(), partW, partB, dw, db, chunks, cols, bias));
                return;
            }
            int chunks = Emitter.ceilDiv(rows, NormKernels.PARAM_CHUNK);
            FloatArray partW = e.scratch(chunks * cols);
            FloatArray partB = e.scratch(chunks * cols);
            int parts = chunks * cols;
            e.task(Emitter.groups(rows), (g, id) -> g.task(id, NormKernels::rowNormBackward, e.context(), dy, x, w, mean, rstd, dx, cols, mode));
            e.task(Emitter.grid1D(parts), (g, id) -> g.task(id, NormKernels::rowNormParamPartials, e.context(), dy, x, mean, rstd, partW, partB, rows, cols, parts));
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
