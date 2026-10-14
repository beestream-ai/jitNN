package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.GemmKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Fused y = act(x W^T + b): one GEMM with the bias and activation in its epilogue (two kernels on
 * Metal). x is [M, K], W is [N, K] (PyTorch layout), b is [N] or absent. iattr = [M, N, K,
 * hasBias, activation] with activation {@link #NONE}, {@link #RELU} or {@link #GELU} (the last
 * only from inference fusion, which needs no gradient).
 */
public final class Linear implements Op {

    static final Linear OP = new Linear();

    static final int NONE = 0;
    static final int RELU = 1;
    static final int GELU = 2;

    private Linear() {
    }

    /**
     * True when this thread's device runs an FP16 Linear of these dimensions on the tensor cores
     * with an FP32 input (MmaGemmKernels.gemmMMAStrideKA32): CUDA, whole 128 x 128 tiles.
     */
    public static boolean readsFp32Input(int m, int n, int k) {
        return Engine.onCuda() && MatMul.tensorCores() && MatMul.wholeTiles(m, n, k);
    }

    public static Tensor apply(Tensor x, Tensor w, Tensor b, boolean relu) {
        if (!(x.dtype() == DType.F32 && w.dtype() == DType.F16)) {
            HostValues.gemmOperands("linear", x, w);
        }
        Checks.rank(x, 2);
        Checks.rank(w, 2);
        if (x.dim(1) != w.dim(1)) {
            throw new IllegalArgumentException("linear: input " + x.shape() + " does not match weight " + w.shape());
        }
        int m = x.dim(0);
        int n = w.dim(0);
        int k = x.dim(1);
        // half-precision inference: FP16 input and weight give an FP16 output
        boolean halfOut = x.dtype() == DType.F16 && w.dtype() == DType.F16 && org.beehive.jitnn.Autocast.halfActivations();
        Tensor y = Tensor.result(Shape.of(m, n), halfOut ? DType.F16 : DType.F32);
        Tensor[] inputs = b == null ? new Tensor[] { x, w } : new Tensor[] { x, w, b };
        if (b != null) {
            Checks.f32(b);
            if (b.numel() != n) {
                throw new IllegalArgumentException("linear: bias " + b.shape() + " for " + n + " outputs");
            }
        }
        Tensor.record(OP, inputs, new Tensor[] { y }, new int[] { m, n, k, b == null ? 0 : 1, relu ? 1 : 0, halfOut ? 1 : 0 }, null, true);
        return y;
    }

    @Override
    public String name() {
        return "linear";
    }

    @Override
    public boolean acceptsHalf() {
        return true;
    }

    @Override
    public void emit(Emitter e, Node node) {
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        int act = node.iattr[4];
        int epilogue = (node.iattr[3] != 0 ? GemmKernels.BIAS : 0) | (act == RELU ? GemmKernels.RELU : 0) | (act == GELU ? GemmKernels.GELU : 0);
        FloatArray bias = node.iattr[3] != 0 ? e.f32(node.in(2)) : null;
        if (node.out(0).dtype() == DType.F16) {
            // FP16 out: cuBLAS writes FP16, or the half GEMM into an FP32 buffer converted with the epilogue
            var xh = e.f16(node.in(0));
            var wh = e.f16(node.in(1));
            var yh = e.f16(node.out(0));
            if (org.beehive.jitnn.natives.Libs.gemmHalfOut(e, xh, wh, yh, bias, m, n, k, 0, 1, epilogue)) {
                return;
            }
            FloatArray tmp = e.scratch(m * n);
            MatMul.emitGemmHalf(e, node.in(0), node.in(1), tmp, e.dummyF32(), m, n, k, 0, 1, 0);
            FloatArray b = bias != null ? bias : e.dummyF32();
            int total = m * n;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LibraryGlueKernels::epilogueToHalf, e.context(), tmp, b, yh, m, n, total, epilogue));
            return;
        }
        if (node.in(0).dtype() == DType.F32 && node.in(1).dtype() == DType.F16) {
            // FP32 input and FP16 weight (Autocast): rounded to FP16 as the tensor-core kernel stages it
            MatMul.emitGemmA32(e, e.f32(node.in(0)), node.in(1), e.f32(node.out(0)), bias != null ? bias : e.dummyF32(), m, n, k, epilogue);
            return;
        }
        if (HostValues.isHalf(node.in(0))) {
            MatMul.emitGemmHalf(e, node.in(0), node.in(1), e.f32(node.out(0)), bias != null ? bias : e.dummyF32(), m, n, k, 0, 1, epilogue);
            return;
        }
        MatMul.emitGemm(e, e.f32(node.in(0)), e.f32(node.in(1)), e.f32(node.out(0)), bias, m, n, k, 0, 1, epilogue);
    }

    @Override
    public void reference(Node node) {
        Object x = node.in(0).storage().host();
        Object w = node.in(1).storage().host();
        FloatArray b = node.iattr[3] != 0 ? (FloatArray) node.in(2).storage().host() : null;
        Object y = node.out(0).storage().host();
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        int act = node.iattr[4];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double acc = b == null ? 0 : b.get(j);
                for (int q = 0; q < k; q++) {
                    acc += (double) HostValues.get(x, i * k + q) * HostValues.get(w, j * k + q);
                }
                double v = act == RELU ? Math.max(acc, 0) : (act == GELU ? acc * RefMath.normalCdf(acc) : acc);
                HostValues.set(y, i * n + j, (float) v);
            }
        }
    }

    @Override
    public Tensor[] backward(Node node, Tensor gy) {
        Tensor x = node.in(0);
        Tensor w = node.in(1);
        if (node.iattr[4] == GELU) {
            throw new UnsupportedOperationException("linear with a fused GELU is for inference only");
        }
        // ReLU gradient from the output: y > 0 exactly where the pre-activation is > 0.
        Tensor g = node.iattr[4] == RELU ? Binary.apply(Binary.Kind.RELU_GRAD, node.out(0), gy) : gy;
        // Half-precision operands: the gradient products run in that type (as under PyTorch's autocast).
        Tensor gl = g.to(x.dtype());
        Tensor gx = x.requiresGrad() ? MatMul.apply(gl, w, false, false) : null;
        Tensor gw = w.requiresGrad() ? MatMul.apply(gl, x, true, false) : null;
        if (node.iattr[3] == 0) {
            return new Tensor[] { gx, gw };
        }
        Tensor b = node.in(2);
        Tensor gb = b.requiresGrad() ? g.sum(0, false).reshape(b.shape().dims()) : null;
        return new Tensor[] { gx, gw, gb };
    }
}
