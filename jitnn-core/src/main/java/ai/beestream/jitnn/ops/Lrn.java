package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.NormKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Local response normalisation across channels of x [N, C, ...], PyTorch's local_response_norm.
 * Outputs y and the denominators' base s (for the gradient). iattr = [C, plane, size]; fattr =
 * [alpha, beta, k].
 */
public final class Lrn implements Op {

    static final Lrn OP = new Lrn();
    static final Backward BACKWARD = new Backward();

    private Lrn() {
    }

    public static Tensor apply(Tensor x, int size, float alpha, float beta, float k) {
        Checks.f32(x);
        if (x.rank() < 3) {
            throw new IllegalArgumentException("local response norm needs [N, C, ...], got " + x.shape());
        }
        int c = x.dim(1);
        int plane = x.numel() / (x.dim(0) * c);
        Tensor y = Tensor.result(x.shape(), DType.F32);
        Tensor s = Tensor.result(x.shape(), DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { y, s }, new int[] { c, plane, size }, new float[] { alpha, beta, k }, true);
        return y;
    }

    @Override
    public String name() {
        return "lrn";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray y = e.f32(n.out(0));
        FloatArray s = e.f32(n.out(1));
        int c = n.iattr[0], plane = n.iattr[1], size = n.iattr[2];
        float alpha = n.fattr[0], beta = n.fattr[1], k = n.fattr[2];
        int total = n.out(0).numel();
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NormKernels::lrnForward, e.context(), x, y, s, c, plane, size, alpha, beta, k, total));
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray y = (FloatArray) n.out(0).storage().host();
        FloatArray s = (FloatArray) n.out(1).storage().host();
        int c = n.iattr[0], plane = n.iattr[1], size = n.iattr[2];
        double alpha = n.fattr[0], beta = n.fattr[1], k = n.fattr[2];
        for (int i = 0; i < n.out(0).numel(); i++) {
            int ch = (i / plane) % c;
            int base = i - ch * plane;
            double acc = 0;
            for (int j = Math.max(0, ch - size / 2); j <= Math.min(c - 1, ch + (size - 1) / 2); j++) {
                acc += (double) x.get(base + j * plane) * x.get(base + j * plane);
            }
            double si = k + alpha / size * acc;
            s.set(i, (float) si);
            y.set(i, (float) (x.get(i) * Math.pow(si, -beta)));
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor dx = Tensor.result(n.in(0).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { gy, n.in(0), n.out(1) }, new Tensor[] { dx }, n.iattr.clone(), n.fattr.clone(), true);
        return new Tensor[] { dx };
    }

    static final class Backward implements Op {

        @Override
        public String name() {
            return "lrn_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray dy = e.f32(n.in(0));
            FloatArray x = e.f32(n.in(1));
            FloatArray s = e.f32(n.in(2));
            FloatArray dx = e.f32(n.out(0));
            int c = n.iattr[0], plane = n.iattr[1], size = n.iattr[2];
            float alpha = n.fattr[0], beta = n.fattr[1];
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, NormKernels::lrnBackward, e.context(), dy, x, s, dx, c, plane, size, alpha, beta, total));
        }

        @Override
        public void reference(Node n) {
            FloatArray dy = (FloatArray) n.in(0).storage().host();
            FloatArray x = (FloatArray) n.in(1).storage().host();
            FloatArray s = (FloatArray) n.in(2).storage().host();
            FloatArray dx = (FloatArray) n.out(0).storage().host();
            int c = n.iattr[0], plane = n.iattr[1], size = n.iattr[2];
            double alpha = n.fattr[0], beta = n.fattr[1];
            for (int i = 0; i < n.out(0).numel(); i++) {
                int ch = (i / plane) % c;
                int base = i - ch * plane;
                double acc = 0;
                for (int j = Math.max(0, ch - (size - 1) / 2); j <= Math.min(c - 1, ch + size / 2); j++) {
                    int at = base + j * plane;
                    acc += (double) dy.get(at) * x.get(at) * Math.pow(s.get(at), -beta - 1);
                }
                dx.set(i, (float) (dy.get(i) * Math.pow(s.get(i), -beta) - 2 * alpha * beta / size * x.get(i) * acc));
            }
        }
    }
}
