package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.ElementwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Element-wise unary operators with up to two scalar parameters. */
public final class Unary implements Op {

    public enum Kind {
        NEG(ElementwiseKernels.NEG),
        EXP(ElementwiseKernels.EXP),
        LOG(ElementwiseKernels.LOG),
        SQRT(ElementwiseKernels.SQRT),
        ABS(ElementwiseKernels.ABS),
        RELU(ElementwiseKernels.RELU),
        AFFINE(ElementwiseKernels.AFFINE),
        POW_SCALAR(ElementwiseKernels.POW_SCALAR),
        CLAMP(ElementwiseKernels.CLAMP),
        RECIPROCAL(ElementwiseKernels.RECIPROCAL),
        SQUARE(ElementwiseKernels.SQUARE),
        FILL(ElementwiseKernels.FILL),
        TANH(ElementwiseKernels.TANH),
        SIGMOID(ElementwiseKernels.SIGMOID),
        GELU(ElementwiseKernels.GELU),
        GELU_TANH(ElementwiseKernels.GELU_TANH),
        SILU(ElementwiseKernels.SILU),
        LEAKY_RELU(ElementwiseKernels.LEAKY_RELU),
        SOFTPLUS(ElementwiseKernels.SOFTPLUS),
        ELU(ElementwiseKernels.ELU),
        SELU(ElementwiseKernels.SELU),
        CELU(ElementwiseKernels.CELU),
        MISH(ElementwiseKernels.MISH),
        SOFTSIGN(ElementwiseKernels.SOFTSIGN),
        HARD_SIGMOID(ElementwiseKernels.HARD_SIGMOID),
        HARD_SWISH(ElementwiseKernels.HARD_SWISH),
        THRESHOLD(ElementwiseKernels.THRESHOLD),
        SIN(ElementwiseKernels.SIN),
        COS(ElementwiseKernels.COS),
        TAN(ElementwiseKernels.TAN),
        ASIN(ElementwiseKernels.ASIN),
        ACOS(ElementwiseKernels.ACOS),
        ATAN(ElementwiseKernels.ATAN),
        SINH(ElementwiseKernels.SINH),
        COSH(ElementwiseKernels.COSH),
        ASINH(ElementwiseKernels.ASINH),
        ACOSH(ElementwiseKernels.ACOSH),
        ATANH(ElementwiseKernels.ATANH),
        ERF(ElementwiseKernels.ERF),
        ERFC(ElementwiseKernels.ERFC),
        ROUND(ElementwiseKernels.ROUND),
        TRUNC(ElementwiseKernels.TRUNC),
        FLOOR(ElementwiseKernels.FLOOR),
        CEIL(ElementwiseKernels.CEIL),
        SIGN(ElementwiseKernels.SIGN),
        LOG1P(ElementwiseKernels.LOG1P),
        EXPM1(ElementwiseKernels.EXPM1),
        LOGICAL_NOT(ElementwiseKernels.LOGICAL_NOT),
        IS_NAN(ElementwiseKernels.IS_NAN),
        IS_INF(ElementwiseKernels.IS_INF),
        IS_FINITE(ElementwiseKernels.IS_FINITE),
        NAN_TO_NUM(ElementwiseKernels.NAN_TO_NUM),
        FAKE_QUANT8(ElementwiseKernels.FAKE_QUANT8),
        FAKE_QUANT16(ElementwiseKernels.FAKE_QUANT16);

        final int code;

        Kind(int code) {
            this.code = code;
        }
    }

    private static final Kind[] BY_CODE = new Kind[Kind.values().length];

    static {
        for (Kind k : Kind.values()) {
            BY_CODE[k.code] = k;
        }
    }

    static final Unary OP = new Unary();

    private Unary() {
    }

    public static Tensor apply(Kind kind, Tensor x, float alpha, float beta) {
        Checks.f32(x);
        // half-precision inference: FP16 in, FP16 out
        Tensor out = Tensor.result(x.shape(), x.dtype() == DType.F16 ? DType.F16 : DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, new int[] { kind.code }, new float[] { alpha, beta }, true);
        return out;
    }

    @Override
    public String name() {
        return "unary";
    }

    @Override
    public boolean acceptsHalf() {
        return true;
    }

    @Override
    public void emit(Emitter e, Node n) {
        if (n.out(0).dtype() == DType.F16) {
            var xh = e.f16(n.in(0));
            var oh = e.f16(n.out(0));
            int code = n.iattr[0];
            float alpha = n.fattr[0];
            float beta = n.fattr[1];
            int count = n.out(0).numel();
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::unaryHalf, e.context(), xh, oh, code, alpha, beta, count));
            return;
        }
        FloatArray x = e.f32(n.in(0));
        FloatArray out = e.f32(n.out(0));
        int code = n.iattr[0];
        float alpha = n.fattr[0];
        float beta = n.fattr[1];
        int count = n.out(0).numel();
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::unary, e.context(), x, out, code, alpha, beta, count));
    }

    @Override
    public void reference(Node n) {
        Object x = n.in(0).storage().host();
        Object o = n.out(0).storage().host();
        for (int i = 0; i < n.out(0).numel(); i++) {
            HostValues.set(o, i, (float) RefMath.unary(n.iattr[0], HostValues.get(x, i), n.fattr[0], n.fattr[1]));
        }
    }

    @Override
    public boolean cpu(Node n) {
        return CpuKernels.unary(n.iattr[0], n.in(0), n.out(0), n.fattr[0], n.fattr[1]);
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor x = n.in(0);
        Tensor y = n.out(0);
        float alpha = n.fattr[0];
        float beta = n.fattr[1];
        Tensor gx = switch (BY_CODE[n.iattr[0]]) {
            case NEG -> gy.neg();
            case EXP -> gy.mul(y);
            case LOG -> gy.div(x);
            case SQRT -> gy.div(y).mul(0.5f);
            case ABS -> Binary.apply(Binary.Kind.SIGN_MUL, x, gy);
            case RELU -> Binary.apply(Binary.Kind.RELU_GRAD, x, gy);
            case AFFINE -> gy.mul(alpha);
            case POW_SCALAR -> gy.mul(x.pow(alpha - 1.0f).mul(alpha));
            case CLAMP -> Binary.apply(Binary.Kind.CLAMP_GRAD, x, gy, alpha, beta);
            case RECIPROCAL -> gy.mul(y.square()).neg();
            case SQUARE -> gy.mul(x.mul(2.0f));
            case FILL -> null;
            case TANH -> Binary.apply(Binary.Kind.TANH_GRAD, y, gy);
            case SIGMOID -> Binary.apply(Binary.Kind.SIGMOID_GRAD, y, gy);
            case GELU -> Binary.apply(Binary.Kind.GELU_GRAD, x, gy);
            case GELU_TANH -> Binary.apply(Binary.Kind.GELU_TANH_GRAD, x, gy);
            case SILU -> Binary.apply(Binary.Kind.SILU_GRAD, x, gy);
            case LEAKY_RELU -> Binary.apply(Binary.Kind.LEAKY_RELU_GRAD, x, gy, alpha, 0.0f);
            case SOFTPLUS -> Binary.apply(Binary.Kind.SOFTPLUS_GRAD, x, gy);
            case ELU -> Binary.apply(Binary.Kind.ELU_GRAD, x, gy, alpha, 0.0f);
            case SELU -> Binary.apply(Binary.Kind.SELU_GRAD, x, gy);
            case CELU -> Binary.apply(Binary.Kind.CELU_GRAD, x, gy, alpha, 0.0f);
            case MISH -> Binary.apply(Binary.Kind.MISH_GRAD, x, gy);
            case SOFTSIGN -> Binary.apply(Binary.Kind.SOFTSIGN_GRAD, x, gy);
            case HARD_SIGMOID -> Binary.apply(Binary.Kind.HARD_SIGMOID_GRAD, x, gy);
            case HARD_SWISH -> Binary.apply(Binary.Kind.HARD_SWISH_GRAD, x, gy);
            case THRESHOLD -> Binary.apply(Binary.Kind.THRESHOLD_GRAD, x, gy, alpha, 0.0f);
            case SIN -> gy.mul(x.cos());
            case COS -> gy.mul(x.sin()).neg();
            case TAN -> gy.mul(y.square().add(1.0f));
            case ASIN -> gy.div(x.square().neg().add(1.0f).sqrt());
            case ACOS -> gy.div(x.square().neg().add(1.0f).sqrt()).neg();
            case ATAN -> gy.div(x.square().add(1.0f));
            case SINH -> gy.mul(x.cosh());
            case COSH -> gy.mul(x.sinh());
            case ASINH -> gy.div(x.square().add(1.0f).sqrt());
            case ACOSH -> gy.div(x.square().add(-1.0f).sqrt());
            case ATANH -> gy.div(x.square().neg().add(1.0f));
            case ERF -> gy.mul(x.square().neg().exp().mul(1.1283791671f));
            case ERFC -> gy.mul(x.square().neg().exp().mul(-1.1283791671f));
            case ROUND -> null;
            case TRUNC -> null;
            case FLOOR -> null;
            case CEIL -> null;
            case SIGN -> null;
            case LOG1P -> gy.div(x.add(1.0f));
            case EXPM1 -> gy.mul(y.add(1.0f));
            case LOGICAL_NOT, IS_NAN, IS_INF, IS_FINITE -> null;
            case NAN_TO_NUM -> gy.mul(x.isFiniteMask());
            case FAKE_QUANT8, FAKE_QUANT16 -> null;
        };
        return new Tensor[] { gx };
    }
}
