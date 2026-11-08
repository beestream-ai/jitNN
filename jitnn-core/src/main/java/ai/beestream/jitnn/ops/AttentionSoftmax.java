package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.SoftmaxKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * softmax(scale * scores) over the last axis of [..., T, S] attention scores, optionally causal
 * (query t sees keys 0 .. t, as PyTorch's is_causal). iattr = [rows, S, causal, T]; fattr = [scale].
 */
public final class AttentionSoftmax implements Op {

    static final AttentionSoftmax OP = new AttentionSoftmax();
    static final Backward BACKWARD = new Backward();

    private AttentionSoftmax() {
    }

    public static Tensor apply(Tensor scores, float scale, boolean causal) {
        Checks.f32(scores);
        if (scores.rank() < 2) {
            throw new IllegalArgumentException("attention scores need [..., T, S], got " + scores.shape());
        }
        int cols = scores.dim(-1);
        int queries = scores.dim(-2);
        int rows = scores.numel() / Math.max(cols, 1);
        Tensor y = Tensor.result(scores.shape(), DType.F32);
        Tensor.record(OP, new Tensor[] { scores }, new Tensor[] { y }, new int[] { rows, cols, causal ? 1 : 0, queries }, new float[] { scale }, true);
        return y;
    }

    @Override
    public String name() {
        return "attention_softmax";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray y = e.f32(n.out(0));
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        int causal = n.iattr[2];
        int queries = n.iattr[3];
        float scale = n.fattr[0];
        e.task(Emitter.groups(rows), (g, id) -> g.task(id, SoftmaxKernels::attentionSoftmaxRows, e.context(), x, y, cols, scale, causal, queries));
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray y = (FloatArray) n.out(0).storage().host();
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        boolean causal = n.iattr[2] != 0;
        int queries = n.iattr[3];
        double scale = n.fattr[0];
        for (int r = 0; r < rows; r++) {
            int valid = causal ? Math.min(cols, r % queries + 1) : cols;
            double m = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < valid; c++) {
                m = Math.max(m, scale * x.get(r * cols + c));
            }
            double s = 0;
            for (int c = 0; c < valid; c++) {
                s += Math.exp(scale * x.get(r * cols + c) - m);
            }
            for (int c = 0; c < cols; c++) {
                y.set(r * cols + c, c < valid ? (float) (Math.exp(scale * x.get(r * cols + c) - m) / s) : 0.0f);
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor dx = Tensor.result(n.in(0).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { n.out(0), gy }, new Tensor[] { dx }, n.iattr.clone(), n.fattr.clone(), true);
        return new Tensor[] { dx };
    }

    static final class Backward implements Op {

        @Override
        public String name() {
            return "attention_softmax_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray y = e.f32(n.in(0));
            FloatArray dy = e.f32(n.in(1));
            FloatArray dx = e.f32(n.out(0));
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            float scale = n.fattr[0];
            e.task(Emitter.groups(rows), (g, id) -> g.task(id, SoftmaxKernels::attentionSoftmaxBackwardRows, e.context(), y, dy, dx, cols, scale));
        }

        @Override
        public void reference(Node n) {
            FloatArray y = (FloatArray) n.in(0).storage().host();
            FloatArray dy = (FloatArray) n.in(1).storage().host();
            FloatArray dx = (FloatArray) n.out(0).storage().host();
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            double scale = n.fattr[0];
            for (int r = 0; r < rows; r++) {
                double dot = 0;
                for (int c = 0; c < cols; c++) {
                    dot += (double) dy.get(r * cols + c) * y.get(r * cols + c);
                }
                for (int c = 0; c < cols; c++) {
                    dx.set(r * cols + c, (float) (scale * y.get(r * cols + c) * (dy.get(r * cols + c) - dot)));
                }
            }
        }
    }
}
