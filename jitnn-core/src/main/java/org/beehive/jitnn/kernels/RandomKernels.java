package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Counter-based random numbers (Philox4x32-10, Salmon et al., SC'11). Element {@code i} depends
 * only on (seed, stream, i), so results are identical on every backend and for any launch shape.
 *
 * <p>WORKAROUND(TornadoVM#1140): no {@code >>>}. The 32x32->64 multiply is done in {@code long}
 * and the high word taken with a signed shift, which is exact once cast back to {@code int}.
 */
public final class RandomKernels {

    public static final int ROUNDS = 10;
    static final int M0 = 0xD2511F53;
    static final int M1 = 0xCD9E8D57;
    static final int W0 = 0x9E3779B9;
    static final int W1 = 0xBB67AE85;
    /** 2^-24. */
    static final float U24 = 5.9604645e-8f;

    private RandomKernels() {
    }

    public static int mulhi(int a, int b) {
        long p = ((long) a & 0xFFFFFFFFL) * ((long) b & 0xFFFFFFFFL);
        return (int) (p >> 32);
    }

    /** Uniform float in [0, 1) from the top 24 bits of a word. */
    public static float toUnit(int w) {
        return ((w >> 8) & 0xFFFFFF) * U24;
    }

    /**
     * Fills {@code out[0, n)}: {@code mode == 0} uniform in [a, b); {@code mode == 1} normal with
     * mean a and standard deviation b (Box-Muller). Element i takes word {@code i & 3} of Philox
     * block {@code i >> 2}; normals pair words (0, 1) and (2, 3).
     *
     * <p>WORKAROUND(TornadoVM, OpenCL): one thread per element. The natural shape, one thread per
     * block writing four values behind nested bounds checks, lets Graal prove an inner check
     * always true, and the OpenCL backend then fails on the constant branch ("logic node
     * (LogicConstantNode)" in emitIf).
     */
    public static void philox(KernelContext kc, FloatArray out, int seedLo, int seedHi, int stream, int mode, float a, float b, int n, int rounds) {
        int i = kc.globalIdx;
        if (i < n) {
            int c0 = i >> 2;
            int c1 = stream;
            int c2 = 0;
            int c3 = 0;
            int k0 = seedLo;
            int k1 = seedHi;
            for (int r = 0; r < rounds; r++) {
                int hi0 = mulhi(M0, c0);
                int lo0 = M0 * c0;
                int hi1 = mulhi(M1, c2);
                int lo1 = M1 * c2;
                c0 = hi1 ^ c1 ^ k0;
                c1 = lo1;
                c2 = hi0 ^ c3 ^ k1;
                c3 = lo0;
                k0 = k0 + W0;
                k1 = k1 + W1;
            }
            int j = i & 3;
            float v;
            if (mode == 0) {
                int w = j == 0 ? c0 : (j == 1 ? c1 : (j == 2 ? c2 : c3));
                v = a + (b - a) * toUnit(w);
            } else {
                int wr = j < 2 ? c0 : c2;
                int wt = j < 2 ? c1 : c3;
                float radius = TornadoMath.sqrt(-2.0f * TornadoMath.log(1.0f - toUnit(wr)));
                float theta = 6.2831855f * toUnit(wt);
                float trig = (j & 1) == 0 ? TornadoMath.cos(theta) : TornadoMath.sin(theta);
                v = a + b * radius * trig;
            }
            out.set(i, v);
        }
    }

    /** Word {@code j} (0..3) of the Philox4x32 block for counter (c0, c1, 0, 0) and key (k0, k1). */
    public static int philoxWord(int c0, int c1, int k0, int k1, int j, int rounds) {
        int a = c0;
        int b = c1;
        int c = 0;
        int d = 0;
        int x = k0;
        int y = k1;
        for (int r = 0; r < rounds; r++) {
            int hi0 = mulhi(M0, a);
            int lo0 = M0 * a;
            int hi1 = mulhi(M1, c);
            int lo1 = M1 * c;
            a = hi1 ^ b ^ x;
            b = lo1;
            c = hi0 ^ d ^ y;
            d = lo0;
            x = x + W0;
            y = y + W1;
        }
        return j == 0 ? a : (j == 1 ? b : (j == 2 ? c : d));
    }

    /**
     * Inverted dropout: keeps element i with probability 1 - p (Philox uniform >= p, keyed by the
     * per-step seed in {@code seed[0..1]} and the layer's stream) and scales it by 1 / (1 - p).
     * The backward pass calls the same kernel on dy: the mask is regenerated, not stored.
     */
    public static void dropout(KernelContext kc, FloatArray x, FloatArray out, uk.ac.manchester.tornado.api.types.arrays.IntArray seed, int stream, float p, float scale, int n, int rounds) {
        int i = kc.globalIdx;
        if (i < n) {
            int w = philoxWord(i >> 2, stream, seed.get(0), seed.get(1), i & 3, rounds);
            out.set(i, toUnit(w) >= p ? x.get(i) * scale : 0.0f);
        }
    }
}
