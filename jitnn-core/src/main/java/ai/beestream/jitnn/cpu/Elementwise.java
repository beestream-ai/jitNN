package ai.beestream.jitnn.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import ai.beestream.jitnn.kernels.ElementwiseKernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Element-wise operators on the CPU, in parallel over the output. Operands broadcast in the
 * repeating form of {@code Binary.repeat}: element i of the output reads element
 * {@code (i / div) % mod} of the operand, so the output splits into runs where each operand is
 * either contiguous (div 1) or one value (div > 1); the common operators run as vector loops over
 * those runs, the others through the device kernels' scalar functions ({@link ElementwiseKernels}).
 */
public final class Elementwise {

    private static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;
    private static final int L = S.length();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat F = ValueLayout.JAVA_FLOAT;
    /** Elements one parallel chunk handles at least. */
    static final int GRAIN = 1 << 15;

    private Elementwise() {
    }

    /** True when {@link #binaryBlock} runs {@code code} in vector lanes. */
    public static boolean vectorBinary(int code) {
        return code == ElementwiseKernels.ADD || code == ElementwiseKernels.SUB || code == ElementwiseKernels.MUL || code == ElementwiseKernels.DIV
                || code == ElementwiseKernels.ADD_RELU || code == ElementwiseKernels.RELU_GRAD || code == ElementwiseKernels.MAX || code == ElementwiseKernels.MIN
                || code == ElementwiseKernels.GELU_TANH_GRAD;
    }

    /**
     * out[i] = op(a[(i / aDiv) % aMod], b[(i / bDiv) % bMod]) for i in [0, n), op a binary
     * {@link ElementwiseKernels} code with its alpha and beta.
     */
    public static void binary(int code, MemorySegment a, int aDiv, int aMod, MemorySegment b, int bDiv, int bMod, MemorySegment out, int n, float alpha, float beta) {
        boolean vector = vectorBinary(code);
        Cpu.parallelFor(n, GRAIN, (from, to) -> {
            int i = from;
            while (i < to) {
                // the run from i on where neither operand changes form
                int end = Math.min(to, runEnd(i, aDiv, aMod));
                end = Math.min(end, runEnd(i, bDiv, bMod));
                int ai = (i / aDiv) % aMod;
                int bi = (i / bDiv) % bMod;
                boolean aConst = aDiv > 1 || aMod == 1;
                boolean bConst = bDiv > 1 || bMod == 1;
                if (vector) {
                    binaryBlock(code, a, ai, aConst, b, bi, bConst, out, i, end - i);
                } else {
                    for (int j = 0; j < end - i; j++) {
                        float x = a.getAtIndex(F, aConst ? ai : ai + j);
                        float y = b.getAtIndex(F, bConst ? bi : bi + j);
                        out.setAtIndex(F, i + j, ElementwiseKernels.binaryOp(code, x, y, alpha, beta));
                    }
                }
                i = end;
            }
        });
    }

    /** The first index after i where an operand of this form moves to another value or wraps around. */
    private static int runEnd(int i, int div, int mod) {
        if (mod == 1) {
            return Integer.MAX_VALUE;
        }
        if (div > 1) {
            return (i / div + 1) * div;
        }
        return (i / mod + 1) * mod;
    }

    /**
     * out[oi + j] = op(a, b) for j in [0, len): an operand is contiguous from its index, or one value
     * when its {@code const} flag is set. op is a {@link #vectorBinary} code.
     */
    public static void binaryBlock(int code, MemorySegment a, long ai, boolean aConst, MemorySegment b, long bi, boolean bConst, MemorySegment out, long oi, int len) {
        FloatVector ac = aConst ? FloatVector.broadcast(S, a.getAtIndex(F, ai)) : null;
        FloatVector bc = bConst ? FloatVector.broadcast(S, b.getAtIndex(F, bi)) : null;
        int vEnd = S.loopBound(len);
        int j = 0;
        for (; j < vEnd; j += L) {
            FloatVector x = aConst ? ac : FloatVector.fromMemorySegment(S, a, (ai + j) * 4, ORDER);
            FloatVector y = bConst ? bc : FloatVector.fromMemorySegment(S, b, (bi + j) * 4, ORDER);
            FloatVector r;
            if (code == ElementwiseKernels.ADD) {
                r = x.add(y);
            } else if (code == ElementwiseKernels.SUB) {
                r = x.sub(y);
            } else if (code == ElementwiseKernels.MUL) {
                r = x.mul(y);
            } else if (code == ElementwiseKernels.DIV) {
                r = x.div(y);
            } else if (code == ElementwiseKernels.ADD_RELU) {
                r = x.add(y).lanewise(VectorOperators.MAX, 0f);
            } else if (code == ElementwiseKernels.RELU_GRAD) {
                r = FloatVector.zero(S).blend(y, x.compare(VectorOperators.GT, 0f));
            } else if (code == ElementwiseKernels.MAX) {
                r = y.blend(x, x.compare(VectorOperators.GT, y));
            } else if (code == ElementwiseKernels.GELU_TANH_GRAD) {
                r = geluTanhGrad(x, y);
            } else {
                r = y.blend(x, x.compare(VectorOperators.LT, y));
            }
            r.intoMemorySegment(out, (oi + j) * 4, ORDER);
        }
        for (; j < len; j++) {
            float x = a.getAtIndex(F, aConst ? ai : ai + j);
            float y = b.getAtIndex(F, bConst ? bi : bi + j);
            out.setAtIndex(F, oi + j, ElementwiseKernels.binaryOp(code, x, y, 0, 0));
        }
    }

    /** True when {@link #unaryBlock} runs {@code code} in vector lanes. */
    public static boolean vectorUnary(int code) {
        return code == ElementwiseKernels.NEG || code == ElementwiseKernels.ABS || code == ElementwiseKernels.RELU || code == ElementwiseKernels.AFFINE
                || code == ElementwiseKernels.CLAMP || code == ElementwiseKernels.SQUARE || code == ElementwiseKernels.RECIPROCAL || code == ElementwiseKernels.SQRT
                || code == ElementwiseKernels.EXP || code == ElementwiseKernels.SIGMOID || code == ElementwiseKernels.SILU || code == ElementwiseKernels.TANH
                || code == ElementwiseKernels.LEAKY_RELU || code == ElementwiseKernels.GELU || code == ElementwiseKernels.GELU_TANH;
    }

    /** out[i] = op(x[i]) for i in [0, n), op a unary {@link ElementwiseKernels} code with its alpha and beta. */
    public static void unary(int code, MemorySegment x, MemorySegment out, int n, float alpha, float beta) {
        boolean vector = vectorUnary(code);
        Cpu.parallelFor(n, GRAIN, (from, to) -> {
            if (vector) {
                unaryBlock(code, x, from, out, from, to - from, alpha, beta);
            } else {
                for (int j = from; j < to; j++) {
                    out.setAtIndex(F, j, ElementwiseKernels.unaryOp(code, x.getAtIndex(F, j), alpha, beta));
                }
            }
        });
    }

    /** out[oOff + j] = op(x[xOff + j]) for j in [0, len), op a {@link #vectorUnary} code. */
    public static void unaryBlock(int code, MemorySegment x, long xOff, MemorySegment out, long oOff, int len, float alpha, float beta) {
        int vEnd = S.loopBound(len);
        int j = 0;
        for (; j < vEnd; j += L) {
            FloatVector v = FloatVector.fromMemorySegment(S, x, (xOff + j) * 4, ORDER);
            FloatVector r;
            if (code == ElementwiseKernels.NEG) {
                r = v.neg();
            } else if (code == ElementwiseKernels.ABS) {
                r = v.abs();
            } else if (code == ElementwiseKernels.RELU) {
                r = v.lanewise(VectorOperators.MAX, 0f);
            } else if (code == ElementwiseKernels.AFFINE) {
                r = v.fma(alpha, beta);
            } else if (code == ElementwiseKernels.CLAMP) {
                r = v.lanewise(VectorOperators.MAX, alpha).lanewise(VectorOperators.MIN, beta);
            } else if (code == ElementwiseKernels.SQUARE) {
                r = v.mul(v);
            } else if (code == ElementwiseKernels.RECIPROCAL) {
                r = FloatVector.broadcast(S, 1f).div(v);
            } else if (code == ElementwiseKernels.SQRT) {
                r = v.sqrt();
            } else if (code == ElementwiseKernels.EXP) {
                r = exp(v);
            } else if (code == ElementwiseKernels.SIGMOID) {
                r = sigmoid(v);
            } else if (code == ElementwiseKernels.SILU) {
                r = v.mul(sigmoid(v));
            } else if (code == ElementwiseKernels.TANH) {
                r = v.lanewise(VectorOperators.TANH);
            } else if (code == ElementwiseKernels.GELU) {
                r = gelu(v);
            } else if (code == ElementwiseKernels.GELU_TANH) {
                r = v.mul(0.5f).mul(geluTanhInner(v).lanewise(VectorOperators.TANH).add(1f));
            } else {
                r = v.mul(alpha).blend(v, v.compare(VectorOperators.GT, 0f));
            }
            r.intoMemorySegment(out, (oOff + j) * 4, ORDER);
        }
        for (; j < len; j++) {
            out.setAtIndex(F, oOff + j, ElementwiseKernels.unaryOp(code, x.getAtIndex(F, xOff + j), alpha, beta));
        }
    }

    /**
     * Exact GELU, x Phi(x) = x erfc(-x / sqrt(2)) / 2, with the complementary error function of the
     * device kernels ({@link ElementwiseKernels#erfc}: Numerical Recipes' Chebyshev fit, fractional
     * error below 1.2e-7) in vector lanes.
     */
    public static FloatVector gelu(FloatVector x) {
        FloatVector u = x.mul(-0.70710678f);
        FloatVector z = u.abs();
        FloatVector t = FloatVector.broadcast(S, 1f).div(z.fma(0.5f, 1f));
        FloatVector poly = t.fma(0.17087277f, -0.82215223f).fma(t, FloatVector.broadcast(S, 1.48851587f)).fma(t, FloatVector.broadcast(S, -1.13520398f))
                .fma(t, FloatVector.broadcast(S, 0.27886807f)).fma(t, FloatVector.broadcast(S, -0.18628806f)).fma(t, FloatVector.broadcast(S, 0.09678418f))
                .fma(t, FloatVector.broadcast(S, 0.37409196f)).fma(t, FloatVector.broadcast(S, 1.00002368f)).fma(t, FloatVector.broadcast(S, -1.26551223f));
        FloatVector r = t.mul(exp(z.mul(z).neg().add(poly)));
        FloatVector erfc = r.blend(FloatVector.broadcast(S, 2f).sub(r), u.compare(VectorOperators.LT, 0f));
        return x.mul(erfc).mul(0.5f);
    }

    /**
     * e^x in vector lanes, within 3e-7 relative of {@code Math.exp}: x = n ln 2 + r with {@code |r| <= ln 2 / 2},
     * a degree-6 polynomial for e^r, 2^n built in the exponent bits; 0 below -87.3, infinity above
     * 88.7, NaN for NaN. About twice as fast as the Vector API's EXP on Apple silicon.
     */
    public static FloatVector exp(FloatVector x) {
        FloatVector c = x.max(-87.3f).min(88.7f);
        // round to nearest by adding and subtracting 1.5 * 2^23
        FloatVector n = c.mul(1.44269504f).add(12582912f).sub(12582912f);
        FloatVector r = n.fma(FloatVector.broadcast(S, -0.693145752f), c).add(n.mul(-1.42860677e-6f));
        FloatVector p = r.fma(1.38888889e-3f, 8.33333333e-3f).fma(r, FloatVector.broadcast(S, 4.16666667e-2f)).fma(r, FloatVector.broadcast(S, 1.66666667e-1f))
                .fma(r, FloatVector.broadcast(S, 0.5f)).fma(r, FloatVector.broadcast(S, 1f)).fma(r, FloatVector.broadcast(S, 1f));
        IntVector bits = ((IntVector) n.convert(VectorOperators.F2I, 0)).add(127).lanewise(VectorOperators.LSHL, 23);
        FloatVector e = p.mul(bits.reinterpretAsFloats());
        e = e.blend(0f, x.compare(VectorOperators.LT, -87.3f));
        e = e.blend(Float.POSITIVE_INFINITY, x.compare(VectorOperators.GT, 88.7f));
        return e.blend(x, x.test(VectorOperators.IS_NAN));
    }

    private static final float SQRT_2_OVER_PI = 0.79788456f;
    private static final float GELU_CUBIC = 0.044715f;

    /** sqrt(2 / pi) (x + 0.044715 x^3), the argument of tanh-approximated GELU's tanh. */
    private static FloatVector geluTanhInner(FloatVector x) {
        return x.mul(x).mul(GELU_CUBIC).add(1f).mul(x).mul(SQRT_2_OVER_PI);
    }

    /** dy times the derivative of tanh-approximated GELU at x. */
    private static FloatVector geluTanhGrad(FloatVector x, FloatVector dy) {
        FloatVector t = geluTanhInner(x).lanewise(VectorOperators.TANH);
        FloatVector inner = x.mul(x).mul(3 * GELU_CUBIC).add(1f).mul(SQRT_2_OVER_PI);
        FloatVector d = t.add(1f).mul(0.5f).add(x.mul(0.5f).mul(t.mul(t).neg().add(1f)).mul(inner));
        return dy.mul(d);
    }

    private static FloatVector sigmoid(FloatVector v) {
        return FloatVector.broadcast(S, 1f).div(exp(v.neg()).add(1f));
    }

    /** sum of x[off + i] for i in [0, n), in vector lanes. */
    public static float sum(MemorySegment x, long off, int n) {
        FloatVector acc = FloatVector.zero(S);
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            acc = acc.add(FloatVector.fromMemorySegment(S, x, (off + i) * 4, ORDER));
        }
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            s += x.getAtIndex(F, off + i);
        }
        return s;
    }

    /** sum of (x[off + i] - mean)^2 for i in [0, n). */
    public static float sumSquares(MemorySegment x, long off, int n, float mean) {
        FloatVector acc = FloatVector.zero(S);
        FloatVector m = FloatVector.broadcast(S, mean);
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            FloatVector d = FloatVector.fromMemorySegment(S, x, (off + i) * 4, ORDER).sub(m);
            acc = d.fma(d, acc);
        }
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float d = x.getAtIndex(F, off + i) - mean;
            s += d * d;
        }
        return s;
    }

    /**
     * y[i] = (x[i] - mean) * scale * w[i] (+ b[i]) for one row of n values (b may be null).
     */
    public static void normalise(MemorySegment x, long xOff, float mean, float scale, MemorySegment w, MemorySegment b, MemorySegment y, long yOff, int n) {
        FloatVector m = FloatVector.broadcast(S, mean);
        FloatVector sc = FloatVector.broadcast(S, scale);
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            FloatVector v = FloatVector.fromMemorySegment(S, x, (xOff + i) * 4, ORDER).sub(m).mul(sc).mul(FloatVector.fromMemorySegment(S, w, (long) i * 4, ORDER));
            if (b != null) {
                v = v.add(FloatVector.fromMemorySegment(S, b, (long) i * 4, ORDER));
            }
            v.intoMemorySegment(y, (yOff + i) * 4, ORDER);
        }
        for (; i < n; i++) {
            float v = (x.getAtIndex(F, xOff + i) - mean) * scale * w.getAtIndex(F, i);
            y.setAtIndex(F, yOff + i, b != null ? v + b.getAtIndex(F, i) : v);
        }
    }

    /**
     * Softmax of one row of n values in place: max, exp(x - max) and its sum, then the scaling.
     * Returns the log-sum-exp, max + log(sum). Entries of -infinity get probability 0.
     */
    public static float softmaxRow(MemorySegment x, long off, int n) {
        FloatVector mx = FloatVector.broadcast(S, Float.NEGATIVE_INFINITY);
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            mx = mx.max(FloatVector.fromMemorySegment(S, x, (off + i) * 4, ORDER));
        }
        float m = mx.reduceLanes(VectorOperators.MAX);
        for (; i < n; i++) {
            m = Math.max(m, x.getAtIndex(F, off + i));
        }
        if (m == Float.NEGATIVE_INFINITY) {
            // every entry masked: no probability anywhere
            for (i = 0; i < n; i++) {
                x.setAtIndex(F, off + i, 0);
            }
            return Float.NEGATIVE_INFINITY;
        }
        FloatVector vm = FloatVector.broadcast(S, m);
        FloatVector acc = FloatVector.zero(S);
        for (i = 0; i < vEnd; i += L) {
            FloatVector e = exp(FloatVector.fromMemorySegment(S, x, (off + i) * 4, ORDER).sub(vm));
            e.intoMemorySegment(x, (off + i) * 4, ORDER);
            acc = acc.add(e);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float e = (float) Math.exp(x.getAtIndex(F, off + i) - m);
            x.setAtIndex(F, off + i, e);
            sum += e;
        }
        float inv = 1 / sum;
        for (i = 0; i < vEnd; i += L) {
            FloatVector.fromMemorySegment(S, x, (off + i) * 4, ORDER).mul(inv).intoMemorySegment(x, (off + i) * 4, ORDER);
        }
        for (; i < n; i++) {
            x.setAtIndex(F, off + i, x.getAtIndex(F, off + i) * inv);
        }
        return m + (float) Math.log(sum);
    }

    /** x[off + i] = x[off + i] * scale for i in [0, n). */
    public static void scale(MemorySegment x, long off, int n, float scale) {
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            FloatVector.fromMemorySegment(S, x, (off + i) * 4, ORDER).mul(scale).intoMemorySegment(x, (off + i) * 4, ORDER);
        }
        for (; i < n; i++) {
            x.setAtIndex(F, off + i, x.getAtIndex(F, off + i) * scale);
        }
    }

    /** y[i] = act(x[i] * scale + shift) for one row of n values; act is ReLU when {@code relu}. */
    public static void affineRow(MemorySegment x, long xOff, float scale, float shift, boolean relu, MemorySegment y, long yOff, int n) {
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            FloatVector v = FloatVector.fromMemorySegment(S, x, (xOff + i) * 4, ORDER).fma(scale, shift);
            if (relu) {
                v = v.lanewise(VectorOperators.MAX, 0f);
            }
            v.intoMemorySegment(y, (yOff + i) * 4, ORDER);
        }
        for (; i < n; i++) {
            float v = x.getAtIndex(F, xOff + i) * scale + shift;
            y.setAtIndex(F, yOff + i, relu ? Math.max(v, 0) : v);
        }
    }

    /** sum of dy[i] * (x[i] - mean) over one row of n values. */
    public static float sumCentredProduct(MemorySegment dy, long dyOff, MemorySegment x, long xOff, float mean, int n) {
        FloatVector acc = FloatVector.zero(S);
        FloatVector m = FloatVector.broadcast(S, mean);
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            FloatVector d = FloatVector.fromMemorySegment(S, x, (xOff + i) * 4, ORDER).sub(m);
            acc = d.fma(FloatVector.fromMemorySegment(S, dy, (dyOff + i) * 4, ORDER), acc);
        }
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            s += dy.getAtIndex(F, dyOff + i) * (x.getAtIndex(F, xOff + i) - mean);
        }
        return s;
    }

    /** out[i] = a * dy[i] + b * x[i] + c over one row of n values. */
    public static void combineRow(MemorySegment dy, long dyOff, float a, MemorySegment x, long xOff, float b, float c, MemorySegment out, long outOff, int n) {
        int vEnd = S.loopBound(n);
        int i = 0;
        FloatVector vc = FloatVector.broadcast(S, c);
        for (; i < vEnd; i += L) {
            FloatVector v = FloatVector.fromMemorySegment(S, x, (xOff + i) * 4, ORDER).fma(b, 0f).add(vc);
            v = FloatVector.fromMemorySegment(S, dy, (dyOff + i) * 4, ORDER).fma(FloatVector.broadcast(S, a), v);
            v.intoMemorySegment(out, (outOff + i) * 4, ORDER);
        }
        for (; i < n; i++) {
            out.setAtIndex(F, outOff + i, a * dy.getAtIndex(F, dyOff + i) + b * x.getAtIndex(F, xOff + i) + c);
        }
    }

    /** dst[dOff + i] += src[sOff + i] for i in [0, n). */
    public static void addRow(MemorySegment src, long sOff, MemorySegment dst, long dOff, int n) {
        int vEnd = S.loopBound(n);
        int i = 0;
        for (; i < vEnd; i += L) {
            FloatVector.fromMemorySegment(S, dst, (dOff + i) * 4, ORDER).add(FloatVector.fromMemorySegment(S, src, (sOff + i) * 4, ORDER)).intoMemorySegment(dst, (dOff + i) * 4, ORDER);
        }
        for (; i < n; i++) {
            dst.setAtIndex(F, dOff + i, dst.getAtIndex(F, dOff + i) + src.getAtIndex(F, sOff + i));
        }
    }
}
