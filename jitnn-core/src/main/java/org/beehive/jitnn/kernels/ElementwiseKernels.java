package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Element-wise kernels. One kernel per arity, selected by an op code: the branch is uniform
 * across a launch, so its cost is negligible next to the memory traffic, and one compiled kernel
 * serves every op of that arity.
 *
 * <p>Kernel code uses non-short-circuit {@code &} and {@code |} in conditions (TornadoVM bug T1:
 * short-circuit operators inside loops can drop a branch).
 */
public final class ElementwiseKernels {

    // Binary op codes: out = f(a, b).
    public static final int ADD = 0;
    public static final int SUB = 1;
    public static final int MUL = 2;
    public static final int DIV = 3;
    public static final int MAX = 4;
    public static final int MIN = 5;
    public static final int POW = 6;
    /** a = x, b = dy: dy where x > 0. */
    public static final int RELU_GRAD = 7;
    /** sign(a) * b. */
    public static final int SIGN_MUL = 8;
    /** a = x, b = dy: dy where alpha <= x <= beta. */
    public static final int CLAMP_GRAD = 9;
    public static final int GT = 10;
    public static final int LT = 11;
    public static final int GE = 12;
    public static final int EQ = 13;
    /** a (b is ignored): broadcast copy. */
    public static final int FIRST = 14;

    // Unary op codes: out = f(x; alpha, beta).
    public static final int NEG = 0;
    public static final int EXP = 1;
    public static final int LOG = 2;
    public static final int SQRT = 3;
    public static final int ABS = 4;
    public static final int RELU = 5;
    /** alpha * x + beta. */
    public static final int AFFINE = 6;
    /** x ^ alpha. */
    public static final int POW_SCALAR = 7;
    /** clamp(x, alpha, beta). */
    public static final int CLAMP = 8;
    public static final int RECIPROCAL = 9;
    public static final int SQUARE = 10;
    /** Fills with alpha. */
    public static final int FILL = 11;

    private ElementwiseKernels() {
    }

    public static float binaryOp(int op, float x, float y, float alpha, float beta) {
        float r;
        if (op == ADD) {
            r = x + y;
        } else if (op == SUB) {
            r = x - y;
        } else if (op == MUL) {
            r = x * y;
        } else if (op == DIV) {
            r = x / y;
        } else if (op == MAX) {
            r = x > y ? x : y;
        } else if (op == MIN) {
            r = x < y ? x : y;
        } else if (op == POW) {
            r = TornadoMath.pow(x, y);
        } else if (op == RELU_GRAD) {
            r = x > 0.0f ? y : 0.0f;
        } else if (op == SIGN_MUL) {
            r = x > 0.0f ? y : (x < 0.0f ? -y : 0.0f);
        } else if (op == CLAMP_GRAD) {
            r = (x >= alpha) & (x <= beta) ? y : 0.0f;
        } else if (op == GT) {
            r = x > y ? 1.0f : 0.0f;
        } else if (op == LT) {
            r = x < y ? 1.0f : 0.0f;
        } else if (op == GE) {
            r = x >= y ? 1.0f : 0.0f;
        } else if (op == EQ) {
            r = x == y ? 1.0f : 0.0f;
        } else {
            r = x;
        }
        return r;
    }

    public static float unaryOp(int op, float x, float alpha, float beta) {
        float r;
        if (op == NEG) {
            r = -x;
        } else if (op == EXP) {
            r = TornadoMath.exp(x);
        } else if (op == LOG) {
            r = TornadoMath.log(x);
        } else if (op == SQRT) {
            r = TornadoMath.sqrt(x);
        } else if (op == ABS) {
            r = TornadoMath.abs(x);
        } else if (op == RELU) {
            r = x > 0.0f ? x : 0.0f;
        } else if (op == AFFINE) {
            r = alpha * x + beta;
        } else if (op == POW_SCALAR) {
            r = TornadoMath.pow(x, alpha);
        } else if (op == CLAMP) {
            r = x < alpha ? alpha : (x > beta ? beta : x);
        } else if (op == RECIPROCAL) {
            r = 1.0f / x;
        } else if (op == SQUARE) {
            r = x * x;
        } else {
            r = alpha;
        }
        return r;
    }

    /** Same-shape binary op, no index arithmetic. */
    public static void binary(KernelContext kc, FloatArray a, FloatArray b, FloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, binaryOp(op, a.get(i), b.get(i), alpha, beta));
        }
    }

    /**
     * Broadcasting binary op. {@code meta} = [rank, outShape[rank], aStrides[rank], bStrides[rank]]
     * where broadcast dimensions have stride 0.
     */
    public static void binaryBroadcast(KernelContext kc, FloatArray a, FloatArray b, FloatArray out, IntArray meta, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int rank = meta.get(0);
            int rem = i;
            int ai = 0;
            int bi = 0;
            for (int d = rank - 1; d >= 0; d--) {
                int size = meta.get(1 + d);
                int c = rem % size;
                rem = rem / size;
                ai += c * meta.get(1 + rank + d);
                bi += c * meta.get(1 + 2 * rank + d);
            }
            out.set(i, binaryOp(op, a.get(ai), b.get(bi), alpha, beta));
        }
    }

    public static void unary(KernelContext kc, FloatArray x, FloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, unaryOp(op, x.get(i), alpha, beta));
        }
    }

    /**
     * Gathers {@code x} into {@code out} with permuted strides. {@code meta} = [rank,
     * outShape[rank], inStride[rank]] where inStride[d] is the input stride of output dimension d.
     */
    public static void permute(KernelContext kc, FloatArray x, FloatArray out, IntArray meta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int rank = meta.get(0);
            int rem = i;
            int src = 0;
            for (int d = rank - 1; d >= 0; d--) {
                int size = meta.get(1 + d);
                int c = rem % size;
                rem = rem / size;
                src += c * meta.get(1 + rank + d);
            }
            out.set(i, x.get(src));
        }
    }
}
