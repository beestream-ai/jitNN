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
        BCE_LOGITS_DX(ElementwiseKernels.BCE_LOGITS_DX);

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
        Tensor out = Tensor.result(outShape, DType.F32);
        Tensor.record(OP, new Tensor[] { a, b }, new Tensor[] { out }, new int[] { kind.code }, new float[] { alpha, beta }, true);
        return out;
    }

    @Override
    public String name() {
        return "binary";
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
        FloatArray fa = e.f32(a);
        FloatArray fb = e.f32(b);
        FloatArray fo = e.f32(out);
        if (a.shape().equals(out.shape()) && b.shape().equals(out.shape())) {
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::binary, e.context(), fa, fb, fo, code, alpha, beta, count));
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
        FloatArray a = (FloatArray) n.in(0).storage().host();
        FloatArray b = (FloatArray) n.in(1).storage().host();
        FloatArray o = (FloatArray) out.storage().host();
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
            o.set(i, (float) RefMath.binary(n.iattr[0], a.get(ai), b.get(bi), n.fattr[0], n.fattr[1]));
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
            case MAX -> new Tensor[] { ga ? gy.mul(a.ge(b)).sumTo(sa) : null, gb ? gy.mul(a.lt(b)).sumTo(sb) : null };
            case MIN -> new Tensor[] { ga ? gy.mul(b.ge(a)).sumTo(sa) : null, gb ? gy.mul(b.lt(a)).sumTo(sb) : null };
            case FIRST -> new Tensor[] { ga ? gy.sumTo(sa) : null, null };
            case BCE -> new Tensor[] { ga ? gy.mul(apply(Kind.BCE_DX, a, b)).sumTo(sa) : null, gb ? gy.mul(apply(Kind.BCE_DY, a, b)).sumTo(sb) : null };
            case BCE_LOGITS -> new Tensor[] { ga ? gy.mul(apply(Kind.BCE_LOGITS_DX, a, b)).sumTo(sa) : null, gb ? gy.mul(a).neg().sumTo(sb) : null };
            default -> throw new UnsupportedOperationException("no gradient for binary " + BY_CODE[n.iattr[0]]);
        };
    }
}
