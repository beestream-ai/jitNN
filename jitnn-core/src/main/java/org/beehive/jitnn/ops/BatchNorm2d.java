package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.NormKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * BatchNorm over the channels of an NCHW tensor, PyTorch semantics. In training it normalises
 * with the batch statistics and updates the persistent running statistics in place; in inference
 * it uses the running statistics. Inputs [x, gamma, beta, runningMean, runningVar]; outputs [y,
 * mean, invstd, runningMean, runningVar] in training, [y] in inference. iattr = [N, C, HW,
 * training, relu]; fattr = [eps, momentum].
 */
public final class BatchNorm2d implements Op {

    static final BatchNorm2d OP = new BatchNorm2d();
    static final Backward BACKWARD = new Backward();

    private BatchNorm2d() {
    }

    public static Tensor apply(Tensor x, Tensor gamma, Tensor beta, Tensor runMean, Tensor runVar, boolean training, float eps, float momentum, boolean relu) {
        Checks.f32(x, gamma, beta, runMean, runVar);
        if (x.rank() != 4 && x.rank() != 2) {
            throw new IllegalArgumentException("batch norm needs [N, C, H, W] or [N, C], got " + x.shape());
        }
        int n = x.dim(0);
        int c = x.dim(1);
        int hw = x.numel() / Math.max(n * c, 1);
        for (Tensor t : new Tensor[] { gamma, beta, runMean, runVar }) {
            if (t.numel() != c) {
                throw new IllegalArgumentException("batch norm: per-channel tensor " + t.shape() + " for " + c + " channels");
            }
        }
        if (!runMean.storage().isPersistent() || !runVar.storage().isPersistent()) {
            throw new IllegalArgumentException("batch norm running statistics must be persistent");
        }
        // half-precision inference: an FP16 input gives an FP16 output (statistics stay FP32)
        boolean half = x.dtype() == DType.F16 && !training && org.beehive.jitnn.Autocast.halfActivations();
        if (x.dtype() == DType.F16 && !half) {
            x = x.to(DType.F32);
        }
        Tensor y = Tensor.result(x.shape(), half ? DType.F16 : DType.F32);
        int[] a = { n, c, hw, training ? 1 : 0, relu ? 1 : 0 };
        float[] f = { eps, momentum };
        Tensor[] in = { x, gamma, beta, runMean, runVar };
        if (training) {
            Tensor mean = Tensor.result(Shape.of(c), DType.F32);
            Tensor invstd = Tensor.result(Shape.of(c), DType.F32);
            Tensor.record(OP, in, new Tensor[] { y, mean, invstd, runMean, runVar }, a, f, true);
        } else {
            Tensor.record(OP, in, new Tensor[] { y }, a, f, true);
        }
        return y;
    }

    @Override
    public String name() {
        return "batch_norm";
    }

    @Override
    public boolean acceptsHalf() {
        return true;
    }

    @Override
    public void emit(Emitter e, Node node) {
        int[] a = node.iattr;
        if (node.out(0).dtype() == DType.F16) {
            int total = a[0] * a[1] * a[2];
            HalfFloatArray xh = e.f16(node.in(0));
            HalfFloatArray yh = e.f16(node.out(0));
            FloatArray gamma = e.f32(node.in(1));
            FloatArray beta = e.f32(node.in(2));
            FloatArray rm = e.f32(node.in(3));
            FloatArray rv = e.f32(node.in(4));
            int c = a[1], hw = a[2], relu = a[4];
            float eps = node.fattr[0];
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NormKernels::batchApplyRunningHalf, e.context(), xh, yh, rm, rv, gamma, beta, c, hw, total, eps, relu));
            return;
        }
        int n = a[0];
        int c = a[1];
        int hw = a[2];
        int relu = a[4];
        int total = n * c * hw;
        float eps = node.fattr[0];
        float momentum = node.fattr[1];
        FloatArray x = e.f32(node.in(0));
        FloatArray gamma = e.f32(node.in(1));
        FloatArray beta = e.f32(node.in(2));
        FloatArray rm = e.f32(node.in(3));
        FloatArray rv = e.f32(node.in(4));
        FloatArray y = e.f32(node.out(0));
        if (a[3] != 0) {
            FloatArray mean = e.f32(node.out(1));
            FloatArray invstd = e.f32(node.out(2));
            e.task(Emitter.groups(c), (g, id) -> g.task(id, NormKernels::batchStats, e.context(), x, mean, invstd, rm, rv, n, c, hw, eps, momentum));
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NormKernels::batchApply, e.context(), x, y, mean, invstd, gamma, beta, c, hw, total, relu));
        } else {
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NormKernels::batchApplyRunning, e.context(), x, y, rm, rv, gamma, beta, c, hw, total, eps, relu));
        }
    }

    @Override
    public void reference(Node node) {
        int[] a = node.iattr;
        int n = a[0];
        int c = a[1];
        int hw = a[2];
        boolean relu = a[4] != 0;
        double eps = node.fattr[0];
        double momentum = node.fattr[1];
        Object x = node.in(0).storage().host();
        FloatArray gamma = (FloatArray) node.in(1).storage().host();
        FloatArray beta = (FloatArray) node.in(2).storage().host();
        FloatArray rm = (FloatArray) node.in(3).storage().host();
        FloatArray rv = (FloatArray) node.in(4).storage().host();
        Object y = node.out(0).storage().host();
        int count = n * hw;
        for (int ch = 0; ch < c; ch++) {
            double mu;
            double is;
            if (a[3] != 0) {
                double s = 0;
                for (int b = 0; b < n; b++) {
                    for (int j = 0; j < hw; j++) {
                        s += HostValues.get(x, (b * c + ch) * hw + j);
                    }
                }
                mu = s / count;
                double sq = 0;
                for (int b = 0; b < n; b++) {
                    for (int j = 0; j < hw; j++) {
                        double d = HostValues.get(x, (b * c + ch) * hw + j) - mu;
                        sq += d * d;
                    }
                }
                double var = sq / count;
                is = 1.0 / Math.sqrt(var + eps);
                ((FloatArray) node.out(1).storage().host()).set(ch, (float) mu);
                ((FloatArray) node.out(2).storage().host()).set(ch, (float) is);
                double unbiased = count > 1 ? sq / (count - 1) : var;
                rm.set(ch, (float) ((1 - momentum) * rm.get(ch) + momentum * mu));
                rv.set(ch, (float) ((1 - momentum) * rv.get(ch) + momentum * unbiased));
            } else {
                mu = rm.get(ch);
                is = 1.0 / Math.sqrt(rv.get(ch) + eps);
            }
            for (int b = 0; b < n; b++) {
                for (int j = 0; j < hw; j++) {
                    int at = (b * c + ch) * hw + j;
                    double v = (HostValues.get(x, at) - mu) * is * gamma.get(ch) + beta.get(ch);
                    HostValues.set(y, at, (float) (relu ? Math.max(v, 0) : v));
                }
            }
        }
    }

    @Override
    public Tensor[] backward(Node node, Tensor gy) {
        if (node.iattr[3] == 0) {
            throw new UnsupportedOperationException("batch norm backward in inference mode");
        }
        Tensor x = node.in(0);
        Tensor gamma = node.in(1);
        Tensor beta = node.in(2);
        int c = node.iattr[1];
        Tensor g = node.iattr[4] != 0 ? Binary.apply(Binary.Kind.RELU_GRAD, node.out(0), gy) : gy;
        Tensor dx = Tensor.result(x.shape(), DType.F32);
        Tensor dgamma = Tensor.result(Shape.of(c), DType.F32);
        Tensor dbeta = Tensor.result(Shape.of(c), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { g, x, node.out(1), node.out(2), gamma }, new Tensor[] { dx, dgamma, dbeta }, node.iattr.clone(), null, true);
        return new Tensor[] { x.requiresGrad() ? dx : null, gamma.requiresGrad() ? dgamma.reshape(gamma.shape().dims()) : null,
                beta.requiresGrad() ? dbeta.reshape(beta.shape().dims()) : null, null, null };
    }

    /** Inputs [dy, x, mean, invstd, gamma]; outputs [dx, dgamma, dbeta]. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "batch_norm_bwd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            int n = node.iattr[0];
            int c = node.iattr[1];
            int hw = node.iattr[2];
            int total = n * c * hw;
            float invCount = 1.0f / (n * hw);
            FloatArray dy = e.f32(node.in(0));
            FloatArray x = e.f32(node.in(1));
            FloatArray mean = e.f32(node.in(2));
            FloatArray invstd = e.f32(node.in(3));
            FloatArray gamma = e.f32(node.in(4));
            FloatArray dx = e.f32(node.out(0));
            FloatArray dgamma = e.f32(node.out(1));
            FloatArray dbeta = e.f32(node.out(2));
            e.task(Emitter.groups(c), (g, id) -> g.task(id, NormKernels::batchBackwardReduce, e.context(), dy, x, mean, invstd, dgamma, dbeta, n, c, hw));
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NormKernels::batchBackwardApply, e.context(), dy, x, mean, invstd, gamma, dgamma, dbeta, dx, c, hw, total, invCount));
        }

        @Override
        public void reference(Node node) {
            int n = node.iattr[0];
            int c = node.iattr[1];
            int hw = node.iattr[2];
            FloatArray dy = (FloatArray) node.in(0).storage().host();
            FloatArray x = (FloatArray) node.in(1).storage().host();
            FloatArray mean = (FloatArray) node.in(2).storage().host();
            FloatArray invstd = (FloatArray) node.in(3).storage().host();
            FloatArray gamma = (FloatArray) node.in(4).storage().host();
            FloatArray dx = (FloatArray) node.out(0).storage().host();
            FloatArray dgamma = (FloatArray) node.out(1).storage().host();
            FloatArray dbeta = (FloatArray) node.out(2).storage().host();
            double count = (double) n * hw;
            for (int ch = 0; ch < c; ch++) {
                double sa = 0;
                double sb = 0;
                for (int b = 0; b < n; b++) {
                    for (int j = 0; j < hw; j++) {
                        int at = (b * c + ch) * hw + j;
                        sa += dy.get(at);
                        sb += dy.get(at) * (x.get(at) - mean.get(ch)) * invstd.get(ch);
                    }
                }
                dbeta.set(ch, (float) sa);
                dgamma.set(ch, (float) sb);
                for (int b = 0; b < n; b++) {
                    for (int j = 0; j < hw; j++) {
                        int at = (b * c + ch) * hw + j;
                        double xhat = (x.get(at) - mean.get(ch)) * invstd.get(ch);
                        dx.set(at, (float) (gamma.get(ch) * invstd.get(ch) * (dy.get(at) - sa / count - xhat * sb / count)));
                    }
                }
            }
        }
    }
}
