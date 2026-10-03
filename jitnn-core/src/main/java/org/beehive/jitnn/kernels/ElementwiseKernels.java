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
    /** a = y = sigmoid(x), b = dy. */
    public static final int SIGMOID_GRAD = 15;
    /** a = y = tanh(x), b = dy. */
    public static final int TANH_GRAD = 16;
    /** a = x, b = dy, for GELU with the exact (erf) form. */
    public static final int GELU_GRAD = 17;
    /** a = x, b = dy, for GELU with the tanh approximation. */
    public static final int GELU_TANH_GRAD = 18;
    /** a = x, b = dy, for SiLU (x * sigmoid(x)). */
    public static final int SILU_GRAD = 19;

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
    public static final int TANH = 12;
    public static final int SIGMOID = 13;
    /** x * Phi(x), Phi the standard normal CDF (PyTorch's approximate='none'). */
    public static final int GELU = 14;
    /** 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) (PyTorch's approximate='tanh'). */
    public static final int GELU_TANH = 15;
    /** x * sigmoid(x), also called swish. */
    public static final int SILU = 16;

    private static final float SQRT_HALF = 0.70710678f;
    private static final float INV_SQRT_2PI = 0.39894228f;
    private static final float SQRT_2_OVER_PI = 0.79788456f;
    private static final float GELU_CUBIC = 0.044715f;

    private ElementwiseKernels() {
    }

    /**
     * Complementary error function, Numerical Recipes' erfcc: Chebyshev fit with a fractional
     * error below 1.2e-7 everywhere. WORKAROUND(J7): TornadoMath has no erf/erfc.
     */
    public static float erfc(float x) {
        float z = TornadoMath.abs(x);
        float t = 1.0f / (1.0f + 0.5f * z);
        float poly = -1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
                + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
        float r = t * TornadoMath.exp(-z * z + poly);
        return x >= 0.0f ? r : 2.0f - r;
    }

    /**
     * Overflow-free tanh, through exp(-2|x|) in (0, 1]. WORKAROUND(J18): Metal's tanh returns NaN
     * for large arguments (e.g. 66).
     */
    public static float tanh(float x) {
        float e = TornadoMath.exp(-2.0f * TornadoMath.abs(x));
        float t = (1.0f - e) / (1.0f + e);
        return x >= 0.0f ? t : -t;
    }

    /** Overflow-free logistic function. */
    public static float sigmoid(float x) {
        float r;
        if (x >= 0.0f) {
            r = 1.0f / (1.0f + TornadoMath.exp(-x));
        } else {
            float e = TornadoMath.exp(x);
            r = e / (1.0f + e);
        }
        return r;
    }

    /** Standard normal CDF, through erfc so it keeps relative precision in the left tail. */
    public static float normalCdf(float x) {
        return 0.5f * erfc(-x * SQRT_HALF);
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
        } else if (op == SIGMOID_GRAD) {
            r = y * x * (1.0f - x);
        } else if (op == TANH_GRAD) {
            r = y * (1.0f - x * x);
        } else if (op == GELU_GRAD) {
            r = y * (normalCdf(x) + x * INV_SQRT_2PI * TornadoMath.exp(-0.5f * x * x));
        } else if (op == GELU_TANH_GRAD) {
            float t = tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
            r = y * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_CUBIC * x * x));
        } else if (op == SILU_GRAD) {
            float s = sigmoid(x);
            r = y * s * (1.0f + x * (1.0f - s));
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
        } else if (op == TANH) {
            r = tanh(x);
        } else if (op == SIGMOID) {
            r = sigmoid(x);
        } else if (op == GELU) {
            r = x * normalCdf(x);
        } else if (op == GELU_TANH) {
            r = 0.5f * x * (1.0f + tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)));
        } else if (op == SILU) {
            r = x * sigmoid(x);
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

    public static void fill(KernelContext kc, FloatArray out, float value, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, value);
        }
    }

    /** out[t, d] = table[index[t], d] for rows of {@code dim} elements; n = tokens * dim. */
    public static void gatherRows(KernelContext kc, FloatArray table, IntArray index, FloatArray out, int dim, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int t = i / dim;
            int d = i - t * dim;
            out.set(i, table.get(index.get(t) * dim + d));
        }
    }

    /**
     * Deterministic segment sum for the embedding gradient: out[v, d] = sum of dy[positions[k], d]
     * for k in offsets[v] .. offsets[v + 1] - 1, in order; n = rows * dim.
     */
    public static void segmentSumRows(KernelContext kc, FloatArray dy, IntArray offsets, IntArray positions, FloatArray out, int dim, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int v = i / dim;
            int d = i - v * dim;
            float acc = 0.0f;
            int end = offsets.get(v + 1);
            for (int k = offsets.get(v); k < end; k++) {
                acc += dy.get(positions.get(k) * dim + d);
            }
            out.set(i, acc);
        }
    }

    /**
     * Copies a block along one axis, with both tensors seen as [outer, length, inner]:
     * out[o, outOff + j, t] = x[o, inOff + j, t] for j < len. {@code n} = outer * len * inner.
     */
    public static void copyBlock(KernelContext kc, FloatArray x, FloatArray out, int len, int inner, int inLen, int outLen, int inOff, int outOff, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int t = i % inner;
            int r = i / inner;
            int j = r % len;
            int o = r / len;
            out.set((o * outLen + outOff + j) * inner + t, x.get((o * inLen + inOff + j) * inner + t));
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

    /** Tile edge of {@link #transposeTiled}; the local tile is padded to TT + 1 columns. */
    public static final int TT = 16;

    /**
     * Batched matrix transpose out[b, c, r] = x[b, r, c] through a padded local-memory tile, so
     * both the reads and the writes are coalesced. 16x16 threads per tile; groups cover
     * ceil(cols/16) x ceil(rows/16) tiles per batch, with the batch folded into the y dimension.
     */
    public static void transposeTiled(KernelContext kc, FloatArray x, FloatArray out, int rows, int cols, int tilesY) {
        float[] tile = kc.allocateFloatLocalArray(TT * (TT + 1));
        int tx = kc.localIdx;
        int ty = kc.localIdy;
        int gy = kc.groupIdy;
        int batch = gy / tilesY;
        int tileRow = gy - batch * tilesY;
        int r = tileRow * TT + ty;
        int c = kc.groupIdx * TT + tx;
        int base = batch * rows * cols;
        if ((r < rows) & (c < cols)) {
            tile[ty * (TT + 1) + tx] = x.get(base + r * cols + c);
        }
        kc.localBarrier();
        int orow = kc.groupIdx * TT + ty;
        int ocol = tileRow * TT + tx;
        if ((orow < cols) & (ocol < rows)) {
            out.set(base + orow * rows + ocol, tile[tx * (TT + 1) + ty]);
        }
    }

    /**
     * Batched transpose with 32x32 tiles and 32x8 threads, each moving four rows (the classic
     * shape for 32-wide SIMD groups). Groups cover ceil(cols/32) x ceil(rows/32) * batch tiles.
     */
    public static void transposeTiled32(KernelContext kc, FloatArray x, FloatArray out, int rows, int cols, int tilesY) {
        float[] tile = kc.allocateFloatLocalArray(32 * 33);
        int tx = kc.localIdx;
        int ty = kc.localIdy;
        int gy = kc.groupIdy;
        int batch = gy / tilesY;
        int tileRow = gy - batch * tilesY;
        int base = batch * rows * cols;
        int c = kc.groupIdx * 32 + tx;
        for (int j = 0; j < 32; j += 8) {
            int r = tileRow * 32 + ty + j;
            if ((r < rows) & (c < cols)) {
                tile[(ty + j) * 33 + tx] = x.get(base + r * cols + c);
            }
        }
        kc.localBarrier();
        int ocol = tileRow * 32 + tx;
        for (int j = 0; j < 32; j += 8) {
            int orow = kc.groupIdx * 32 + ty + j;
            if ((orow < cols) & (ocol < rows)) {
                out.set(base + orow * rows + ocol, tile[tx * 33 + ty + j]);
            }
        }
    }
}
