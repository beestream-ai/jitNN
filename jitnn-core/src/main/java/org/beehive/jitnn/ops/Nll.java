package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.SoftmaxKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Per-row negative log-likelihood: loss[r] = -logp[r, labels[r]]. iattr = [rows, cols].
 * Combined with log-softmax and a mean it is cross-entropy.
 */
public final class Nll implements Op {

    static final Nll OP = new Nll();
    static final Backward BACKWARD = new Backward();

    private Nll() {
    }

    public static Tensor apply(Tensor logp, Tensor labels) {
        Checks.f32(logp);
        Checks.rank(logp, 2);
        if (labels.dtype() != DType.I32 || labels.rank() != 1 || labels.dim(0) != logp.dim(0)) {
            throw new IllegalArgumentException("labels must be INT32 of shape [" + logp.dim(0) + "], got " + labels);
        }
        int rows = logp.dim(0);
        int cols = logp.dim(1);
        Tensor loss = Tensor.result(Shape.of(rows), DType.F32);
        Tensor.record(OP, new Tensor[] { logp, labels }, new Tensor[] { loss }, new int[] { rows, cols }, null, true);
        return loss;
    }

    @Override
    public String name() {
        return "nll";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray logp = e.f32(n.in(0));
        IntArray labels = e.i32(n.in(1));
        FloatArray loss = e.f32(n.out(0));
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, SoftmaxKernels::nllRows, e.context(), logp, labels, loss, rows, cols));
    }

    @Override
    public void reference(Node n) {
        FloatArray logp = (FloatArray) n.in(0).storage().host();
        IntArray labels = (IntArray) n.in(1).storage().host();
        FloatArray loss = (FloatArray) n.out(0).storage().host();
        int cols = n.iattr[1];
        for (int r = 0; r < n.iattr[0]; r++) {
            int c = labels.get(r);
            if (c < 0 || c >= cols) {
                throw new IllegalArgumentException("label " + c + " out of range [0, " + cols + ")");
            }
            loss.set(r, -logp.get(r * cols + c));
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor dlogp = Tensor.result(n.in(0).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { n.in(1), gy }, new Tensor[] { dlogp }, n.iattr.clone(), null, true);
        return new Tensor[] { dlogp, null };
    }

    static final class Backward implements Op {

        @Override
        public String name() {
            return "nll_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray labels = e.i32(n.in(0));
            FloatArray dloss = e.f32(n.in(1));
            FloatArray dlogp = e.f32(n.out(0));
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            e.task(Emitter.grid1D(rows * cols), (g, id) -> g.task(id, SoftmaxKernels::nllBackward, e.context(), labels, dloss, dlogp, rows, cols));
        }

        @Override
        public void reference(Node n) {
            IntArray labels = (IntArray) n.in(0).storage().host();
            FloatArray dloss = (FloatArray) n.in(1).storage().host();
            FloatArray dlogp = (FloatArray) n.out(0).storage().host();
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    dlogp.set(r * cols + c, c == labels.get(r) ? -dloss.get(r) : 0.0f);
                }
            }
        }
    }
}
