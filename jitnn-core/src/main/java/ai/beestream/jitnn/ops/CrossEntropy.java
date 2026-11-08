package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.ReduceKernels;
import ai.beestream.jitnn.kernels.SoftmaxKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Fused mean cross-entropy of logits [N, C] and INT32 labels [N]: a row kernel plus a mean
 * forward, and a single kernel backward ((softmax - one-hot) * g / N) instead of log-softmax, NLL
 * and mean with their three gradients. iattr = [rows, cols].
 */
public final class CrossEntropy implements Op {

    static final CrossEntropy OP = new CrossEntropy();
    static final Backward BACKWARD = new Backward();

    private CrossEntropy() {
    }

    public static Tensor apply(Tensor logits, Tensor labels) {
        Checks.f32(logits);
        Checks.rank(logits, 2);
        if (labels.dtype() != DType.I32 || labels.rank() != 1 || labels.dim(0) != logits.dim(0)) {
            throw new IllegalArgumentException("labels must be INT32 of shape [" + logits.dim(0) + "], got " + labels);
        }
        Tensor loss = Tensor.result(Shape.scalar(), DType.F32);
        Tensor.record(OP, new Tensor[] { logits, labels }, new Tensor[] { loss }, new int[] { logits.dim(0), logits.dim(1) }, null, true);
        return loss;
    }

    @Override
    public String name() {
        return "cross_entropy";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        IntArray labels = e.i32(n.in(1));
        FloatArray out = e.f32(n.out(0));
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        FloatArray rowLoss = e.scratch(rows);
        float scale = 1.0f / rows;
        e.task(Emitter.groups(rows), (g, id) -> g.task(id, SoftmaxKernels::crossEntropyRows, e.context(), x, labels, rowLoss, cols));
        e.task(Emitter.groups(1), (g, id) -> g.task(id, ReduceKernels::reduceGroup, e.context(), rowLoss, out, 1, rows, 1, ReduceKernels.SUM, scale));
    }

    @Override
    public boolean cpu(Node n) {
        return CpuKernels.crossEntropy(n.in(0), n.in(1), n.out(0), n.iattr[0], n.iattr[1]);
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        IntArray labels = (IntArray) n.in(1).storage().host();
        FloatArray out = (FloatArray) n.out(0).storage().host();
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        double total = 0;
        for (int r = 0; r < rows; r++) {
            double m = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                m = Math.max(m, x.get(r * cols + c));
            }
            double s = 0;
            for (int c = 0; c < cols; c++) {
                s += Math.exp(x.get(r * cols + c) - m);
            }
            int label = labels.get(r);
            if (label < 0 || label >= cols) {
                throw new IllegalArgumentException("label " + label + " out of range [0, " + cols + ")");
            }
            total += Math.log(s) + m - x.get(r * cols + label);
        }
        out.set(0, (float) (total / rows));
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor dx = Tensor.result(n.in(0).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { n.in(0), n.in(1), gy }, new Tensor[] { dx }, n.iattr.clone(), null, true);
        return new Tensor[] { dx, null };
    }

    static final class Backward implements Op {

        @Override
        public String name() {
            return "cross_entropy_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            IntArray labels = e.i32(n.in(1));
            FloatArray g = e.f32(n.in(2));
            FloatArray dx = e.f32(n.out(0));
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            float scale = 1.0f / rows;
            e.task(Emitter.groups(rows), (tg, id) -> tg.task(id, SoftmaxKernels::crossEntropyBackwardRows, e.context(), x, labels, g, dx, cols, scale));
        }

        @Override
        public boolean cpu(Node n) {
            return CpuKernels.crossEntropyBackward(n.in(0), n.in(1), n.in(2), n.out(0), n.iattr[0], n.iattr[1]);
        }

        @Override
        public void reference(Node n) {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            IntArray labels = (IntArray) n.in(1).storage().host();
            FloatArray g = (FloatArray) n.in(2).storage().host();
            FloatArray dx = (FloatArray) n.out(0).storage().host();
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            double gs = g.get(0) / (double) rows;
            for (int r = 0; r < rows; r++) {
                double m = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < cols; c++) {
                    m = Math.max(m, x.get(r * cols + c));
                }
                double s = 0;
                for (int c = 0; c < cols; c++) {
                    s += Math.exp(x.get(r * cols + c) - m);
                }
                for (int c = 0; c < cols; c++) {
                    double p = Math.exp(x.get(r * cols + c) - m) / s;
                    dx.set(r * cols + c, (float) ((p - (c == labels.get(r) ? 1 : 0)) * gs));
                }
            }
        }
    }
}
