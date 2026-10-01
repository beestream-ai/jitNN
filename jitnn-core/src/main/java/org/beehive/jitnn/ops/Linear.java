package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.GemmKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Fused y = act(x W^T + b): one GEMM with the bias and activation in its epilogue (two kernels on
 * Metal). x is [M, K], W is [N, K] (PyTorch layout), b is [N] or absent. iattr = [M, N, K,
 * hasBias, relu].
 */
public final class Linear implements Op {

    static final Linear OP = new Linear();

    private Linear() {
    }

    public static Tensor apply(Tensor x, Tensor w, Tensor b, boolean relu) {
        Checks.f32(x, w);
        Checks.rank(x, 2);
        Checks.rank(w, 2);
        if (x.dim(1) != w.dim(1)) {
            throw new IllegalArgumentException("linear: input " + x.shape() + " does not match weight " + w.shape());
        }
        int m = x.dim(0);
        int n = w.dim(0);
        int k = x.dim(1);
        Tensor y = Tensor.result(Shape.of(m, n), DType.F32);
        Tensor[] inputs = b == null ? new Tensor[] { x, w } : new Tensor[] { x, w, b };
        if (b != null) {
            Checks.f32(b);
            if (b.numel() != n) {
                throw new IllegalArgumentException("linear: bias " + b.shape() + " for " + n + " outputs");
            }
        }
        Tensor.record(OP, inputs, new Tensor[] { y }, new int[] { m, n, k, b == null ? 0 : 1, relu ? 1 : 0 }, null, true);
        return y;
    }

    @Override
    public String name() {
        return "linear";
    }

    @Override
    public void emit(Emitter e, Node node) {
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        int epilogue = (node.iattr[3] != 0 ? GemmKernels.BIAS : 0) | (node.iattr[4] != 0 ? GemmKernels.RELU : 0);
        FloatArray bias = node.iattr[3] != 0 ? e.f32(node.in(2)) : null;
        MatMul.emitGemm(e, e.f32(node.in(0)), e.f32(node.in(1)), e.f32(node.out(0)), bias, m, n, k, 0, 1, epilogue);
    }

    @Override
    public void reference(Node node) {
        FloatArray x = (FloatArray) node.in(0).storage().host();
        FloatArray w = (FloatArray) node.in(1).storage().host();
        FloatArray b = node.iattr[3] != 0 ? (FloatArray) node.in(2).storage().host() : null;
        FloatArray y = (FloatArray) node.out(0).storage().host();
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        boolean relu = node.iattr[4] != 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double acc = b == null ? 0 : b.get(j);
                for (int q = 0; q < k; q++) {
                    acc += (double) x.get(i * k + q) * w.get(j * k + q);
                }
                y.set(i * n + j, (float) (relu ? Math.max(acc, 0) : acc));
            }
        }
    }

    @Override
    public Tensor[] backward(Node node, Tensor gy) {
        Tensor x = node.in(0);
        Tensor w = node.in(1);
        // ReLU gradient from the output: y > 0 exactly where the pre-activation is > 0.
        Tensor g = node.iattr[4] != 0 ? Binary.apply(Binary.Kind.RELU_GRAD, node.out(0), gy) : gy;
        Tensor gx = x.requiresGrad() ? MatMul.apply(g, w, false, false) : null;
        Tensor gw = w.requiresGrad() ? MatMul.apply(g, x, true, false) : null;
        if (node.iattr[3] == 0) {
            return new Tensor[] { gx, gw };
        }
        Tensor b = node.in(2);
        Tensor gb = b.requiresGrad() ? g.sum(0, false).reshape(b.shape().dims()) : null;
        return new Tensor[] { gx, gw, gb };
    }
}
