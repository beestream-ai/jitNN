package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.ElementwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Element-wise binary operators with NumPy broadcasting. */
public final class Binary implements Op {

    public enum Kind {
        ADD(ElementwiseKernels.ADD),
        SUB(ElementwiseKernels.SUB),
        MUL(ElementwiseKernels.MUL),
        DIV(ElementwiseKernels.DIV),
        MAX(ElementwiseKernels.MAX),
        MIN(ElementwiseKernels.MIN),
        POW(ElementwiseKernels.POW),
        RELU_GRAD(ElementwiseKernels.RELU_GRAD),
        SIGN_MUL(ElementwiseKernels.SIGN_MUL),
        CLAMP_GRAD(ElementwiseKernels.CLAMP_GRAD),
        GT(ElementwiseKernels.GT),
        LT(ElementwiseKernels.LT),
        GE(ElementwiseKernels.GE),
        EQ(ElementwiseKernels.EQ),
        FIRST(ElementwiseKernels.FIRST),
        SIGMOID_GRAD(ElementwiseKernels.SIGMOID_GRAD),
        TANH_GRAD(ElementwiseKernels.TANH_GRAD),
        GELU_GRAD(ElementwiseKernels.GELU_GRAD),
        GELU_TANH_GRAD(ElementwiseKernels.GELU_TANH_GRAD),
        SILU_GRAD(ElementwiseKernels.SILU_GRAD),
        LEAKY_RELU_GRAD(ElementwiseKernels.LEAKY_RELU_GRAD),
        SOFTPLUS_GRAD(ElementwiseKernels.SOFTPLUS_GRAD),
        BCE(ElementwiseKernels.BCE),
        BCE_DX(ElementwiseKernels.BCE_DX),
        BCE_DY(ElementwiseKernels.BCE_DY),
        BCE_LOGITS(ElementwiseKernels.BCE_LOGITS),
        BCE_LOGITS_DX(ElementwiseKernels.BCE_LOGITS_DX),
        ADD_RELU(ElementwiseKernels.ADD_RELU),
        ELU_GRAD(ElementwiseKernels.ELU_GRAD),
        SELU_GRAD(ElementwiseKernels.SELU_GRAD),
        CELU_GRAD(ElementwiseKernels.CELU_GRAD),
        MISH_GRAD(ElementwiseKernels.MISH_GRAD),
        SOFTSIGN_GRAD(ElementwiseKernels.SOFTSIGN_GRAD),
        HARD_SIGMOID_GRAD(ElementwiseKernels.HARD_SIGMOID_GRAD),
        HARD_SWISH_GRAD(ElementwiseKernels.HARD_SWISH_GRAD),
        THRESHOLD_GRAD(ElementwiseKernels.THRESHOLD_GRAD),
        PRELU(ElementwiseKernels.PRELU),
        PRELU_SLOPE(ElementwiseKernels.PRELU_SLOPE),
        ATAN2(ElementwiseKernels.ATAN2),
        NE(ElementwiseKernels.NE),
        LE(ElementwiseKernels.LE),
        LOGICAL_AND(ElementwiseKernels.LOGICAL_AND),
        LOGICAL_OR(ElementwiseKernels.LOGICAL_OR),
        LOGICAL_XOR(ElementwiseKernels.LOGICAL_XOR),
        MAXIMUM(ElementwiseKernels.MAXIMUM),
        MINIMUM(ElementwiseKernels.MINIMUM),
        FMAX(ElementwiseKernels.FMAX),
        FMIN(ElementwiseKernels.FMIN),
        REMAINDER(ElementwiseKernels.REMAINDER),
        FMOD(ElementwiseKernels.FMOD);

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

    static final Binary OP = new Binary();

    private Binary() {
    }

    public static Tensor apply(Kind kind, Tensor a, Tensor b) {
        return apply(kind, a, b, Shape.broadcast(a.shape(), b.shape()), 0.0f, 0.0f);
    }

    public static Tensor apply(Kind kind, Tensor a, Tensor b, Shape out) {
        return apply(kind, a, b, out, 0.0f, 0.0f);
    }

    public static Tensor apply(Kind kind, Tensor a, Tensor b, float alpha, float beta) {
        return apply(kind, a, b, Shape.broadcast(a.shape(), b.shape()), alpha, beta);
    }

    static Tensor apply(Kind kind, Tensor a, Tensor b, Shape outShape, float alpha, float beta) {
        Checks.f32(a, b);
        // half-precision inference: FP16 operands of the output's shape give FP16; others are computed in FP32
        boolean half = a.dtype() == DType.F16 && b.dtype() == DType.F16 && a.shape().equals(outShape) && b.shape().equals(outShape);
        if (!half) {
            a = a.dtype() == DType.F16 ? a.to(DType.F32) : a;
            b = b.dtype() == DType.F16 ? b.to(DType.F32) : b;
        }
        Tensor out = Tensor.result(outShape, half ? DType.F16 : DType.F32);
        Tensor.record(OP, new Tensor[] { a, b }, new Tensor[] { out }, new int[] { kind.code }, new float[] { alpha, beta }, true);
        return out;
    }

    /** The {@code (i / div) % mod} kernel for block-repeating broadcasts unless {@code -Djitnn.binaryRepeat=false}. */
    static final boolean REPEAT = !"false".equals(System.getProperty("jitnn.binaryRepeat"));

    /**
     * {div, mod} when the operand, aligned to the output's trailing dimensions, is 1 everywhere
     * except one contiguous block of dimensions equal to the output's: element i of the output
     * reads element (i / div) % mod. Null otherwise.
     */
    static int[] repeat(Shape operand, Shape out) {
        int[] o = out.dims();
        int[] p = operand.dims();
        int lead = o.length - p.length;
        if (lead < 0) {
            return null;
        }
        int first = -1;
        int last = -1;
        for (int d = 0; d < o.length; d++) {
            int size = d < lead ? 1 : p[d - lead];
            if (size != 1) {
                if (size != o[d] || (last >= 0 && last != d - 1)) {
                    return null;
                }
                if (first < 0) {
                    first = d;
                }
                last = d;
            }
        }
        if (first < 0) {
            return new int[] { 1, 1 };
        }
        int mod = 1;
        for (int d = first; d <= last; d++) {
            mod *= o[d];
        }
        int div = 1;
        for (int d = last + 1; d < o.length; d++) {
            div *= o[d];
        }
        return new int[] { div, mod };
    }

    @Override
    public String name() {
        return "binary";
    }

    @Override
    public boolean acceptsHalf() {
        return true;
    }

    @Override
    public void emit(Emitter e, Node n) {
        Tensor a = n.in(0);
        Tensor b = n.in(1);
        Tensor out = n.out(0);
        int code = n.iattr[0];
        float alpha = n.fattr[0];
        float beta = n.fattr[1];
        int count = out.numel();
        if (out.dtype() == DType.F16) {
            var ha = e.f16(a);
            var hb = e.f16(b);
            var ho = e.f16(out);
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::binaryHalf, e.context(), ha, hb, ho, code, alpha, beta, count));
            return;
        }
        FloatArray fa = e.f32(a);
        FloatArray fb = e.f32(b);
        FloatArray fo = e.f32(out);
        if (a.shape().equals(out.shape()) && b.shape().equals(out.shape())) {
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::binary, e.context(), fa, fb, fo, code, alpha, beta, count));
        } else if (REPEAT && repeat(a.shape(), out.shape()) != null && repeat(b.shape(), out.shape()) != null) {
            int[] ra = repeat(a.shape(), out.shape());
            int[] rb = repeat(b.shape(), out.shape());
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::binaryRepeat, e.context(), fa, fb, fo, code, alpha, beta, count, ra[0], ra[1], rb[0], rb[1]));
        } else {
            IntArray meta = e.constant(broadcastMeta(a.shape(), b.shape(), out.shape()));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::binaryBroadcast, e.context(), fa, fb, fo, meta, code, alpha, beta, count));
        }
    }

    static int[] broadcastMeta(Shape a, Shape b, Shape out) {
        Shape o = out.rank() == 0 ? Shape.of(1) : out;
        int r = o.rank();
        int[] sa = a.rank() == 0 ? new int[r] : a.broadcastStrides(o);
        int[] sb = b.rank() == 0 ? new int[r] : b.broadcastStrides(o);
        int[] meta = new int[1 + 3 * r];
        meta[0] = r;
        int[] dims = o.dims();
        for (int d = 0; d < r; d++) {
            meta[1 + d] = dims[d];
            meta[1 + r + d] = sa[d];
            meta[1 + 2 * r + d] = sb[d];
        }
        return meta;
    }

    @Override
    public void reference(Node n) {
        Tensor out = n.out(0);
        int[] meta = broadcastMeta(n.in(0).shape(), n.in(1).shape(), out.shape());
        Object a = n.in(0).storage().host();
        Object b = n.in(1).storage().host();
        Object o = out.storage().host();
        int r = meta[0];
        for (int i = 0; i < out.numel(); i++) {
            int rem = i;
            int ai = 0;
            int bi = 0;
            for (int d = r - 1; d >= 0; d--) {
                int c = rem % meta[1 + d];
                rem /= meta[1 + d];
                ai += c * meta[1 + r + d];
                bi += c * meta[1 + 2 * r + d];
            }
            HostValues.set(o, i, (float) RefMath.binary(n.iattr[0], HostValues.get(a, ai), HostValues.get(b, bi), n.fattr[0], n.fattr[1]));
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor a = n.in(0);
        Tensor b = n.in(1);
        Shape sa = a.shape();
        Shape sb = b.shape();
        boolean ga = a.requiresGrad();
        boolean gb = b.requiresGrad();
        return switch (BY_CODE[n.iattr[0]]) {
            case ADD -> new Tensor[] { ga ? gy.sumTo(sa) : null, gb ? gy.sumTo(sb) : null };
            case SUB -> new Tensor[] { ga ? gy.sumTo(sa) : null, gb ? gy.neg().sumTo(sb) : null };
            case MUL -> new Tensor[] { ga ? gy.mul(b).sumTo(sa) : null, gb ? gy.mul(a).sumTo(sb) : null };
            case DIV -> new Tensor[] { ga ? gy.div(b).sumTo(sa) : null, gb ? gy.mul(a).div(b.square()).neg().sumTo(sb) : null };
            case MAXIMUM, FMAX -> new Tensor[] { ga ? gy.mul(a.ge(b)).sumTo(sa) : null, gb ? gy.mul(a.lt(b)).sumTo(sb) : null };
            case MINIMUM, FMIN -> new Tensor[] { ga ? gy.mul(b.ge(a)).sumTo(sa) : null, gb ? gy.mul(b.lt(a)).sumTo(sb) : null };
            case MAX -> new Tensor[] { ga ? gy.mul(a.ge(b)).sumTo(sa) : null, gb ? gy.mul(a.lt(b)).sumTo(sb) : null };
            case MIN -> new Tensor[] { ga ? gy.mul(b.ge(a)).sumTo(sa) : null, gb ? gy.mul(b.lt(a)).sumTo(sb) : null };
            case FIRST -> new Tensor[] { ga ? gy.sumTo(sa) : null, null };
            case BCE -> new Tensor[] { ga ? gy.mul(apply(Kind.BCE_DX, a, b)).sumTo(sa) : null, gb ? gy.mul(apply(Kind.BCE_DY, a, b)).sumTo(sb) : null };
            case PRELU -> new Tensor[] { ga ? gy.mul(apply(Kind.PRELU_SLOPE, a, b)).sumTo(sa) : null, gb ? gy.mul(a.minimum(Tensor.scalar(0.0f))).sumTo(sb) : null };
            case ATAN2 -> {
                // d atan2(a, b) = (b da - a db) / (a^2 + b^2)
                Tensor r = a.square().add(b.square());
                yield new Tensor[] { ga ? gy.mul(b).div(r).sumTo(sa) : null, gb ? gy.mul(a).div(r).neg().sumTo(sb) : null };
            }
            // d/da = 1; d/db = -floor(a / b) (remainder) or -trunc(a / b) (fmod)
            case REMAINDER -> new Tensor[] { ga ? gy.sumTo(sa) : null, gb ? gy.mul(a.div(b).floor()).neg().sumTo(sb) : null };
            case FMOD -> new Tensor[] { ga ? gy.sumTo(sa) : null, gb ? gy.mul(a.div(b).abs().floor().mul(a.div(b).sign())).neg().sumTo(sb) : null };
            case GT, LT, GE, EQ, NE, LE, LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR -> new Tensor[] { null, null };
            case ADD_RELU -> {
                Tensor g = apply(Kind.RELU_GRAD, n.out(0), gy);
                yield new Tensor[] { ga ? g.sumTo(sa) : null, gb ? g.sumTo(sb) : null };
            }
            case BCE_LOGITS -> new Tensor[] { ga ? gy.mul(apply(Kind.BCE_LOGITS_DX, a, b)).sumTo(sa) : null, gb ? gy.mul(a).neg().sumTo(sb) : null };
            default -> throw new UnsupportedOperationException("no gradient for binary " + BY_CODE[n.iattr[0]]);
        };
    }
}
