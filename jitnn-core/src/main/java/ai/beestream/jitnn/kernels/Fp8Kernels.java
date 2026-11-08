package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.FP8;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * FP8 (OCP E4M3 and E5M2) as a storage type with a per-tensor scale: value = decode(byte) * scale.
 * Encoding uses TornadoVM's {@link FP8} conversion (integer and float arithmetic only, so it runs
 * on every backend); decoding is jitNN's own, building the power of two from its exponent bits.
 * The GEMMs decode their FP8 operands while loading tiles into local memory and accumulate in FP32.
 *
 * <p>WORKAROUND(TornadoVM): the OpenCL backend emits {@code Float.NaN} as {@code NaNF}, which
 * does not compile (scalar kernel arguments are specialised into literals too), so NaN and
 * infinity reach the decoder through a two-element array, special = [NaN, +infinity].
 */
public final class Fp8Kernels {

    public static final int E4M3 = 0;
    public static final int E5M2 = 1;
    public static final int TILE = 16;

    private Fp8Kernels() {
    }

    /** 2^e for e in the normal float range, from the exponent bits. */
    private static float pow2(int e) {
        return Float.intBitsToFloat((e + 127) * 8388608);
    }

    /** The value of FP8 byte {@code b}; nan and inf are Float.NaN and Float.POSITIVE_INFINITY. */
    public static float decode(byte b, int format, float nan, float inf) {
        int u = b;
        if (u < 0) {
            u = u + 256;
        }
        int sign = u / 128;
        int rem = u - sign * 128;
        int mantBits = format == E4M3 ? 8 : 4;
        int bias = format == E4M3 ? 7 : 15;
        int exp = rem / mantBits;
        int mant = rem - exp * mantBits;
        float sig = 1.0f - 2.0f * sign;
        float frac = (float) mant / mantBits;
        float v;
        if (exp == 0) {
            v = sig * frac * pow2(1 - bias);
        } else {
            v = sig * (1.0f + frac) * pow2(exp - bias);
        }
        if (format == E4M3) {
            if ((exp == 15) & (mant == 7)) {
                v = nan;
            }
        } else if (exp == 31) {
            if (mant == 0) {
                v = sig * inf;
            } else {
                v = nan;
            }
        }
        return v;
    }

    public static byte encode(float v, int format) {
        byte b;
        if (format == E4M3) {
            b = FP8.e4m3FromFloat(v);
        } else {
            b = FP8.e5m2FromFloat(v);
        }
        return b;
    }

    /** q = encode(x / scale[0]), saturating at the format's largest finite value. */
    public static void quantize(KernelContext kc, FloatArray x, FloatArray scale, ByteArray q, int format, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            q.set(i, encode(x.get(i) / scale.get(0), format));
        }
    }

    public static void dequantize(KernelContext kc, ByteArray q, FloatArray scale, FloatArray special, FloatArray y, int format, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, decode(q.get(i), format, special.get(0), special.get(1)) * scale.get(0));
        }
    }

    /**
     * y [rows, m] = x [rows, k] (FP32, or FP8 when aFormat >= 0, scaled by sa) times w^T, w [m, k]
     * FP8 scaled by sw, plus bias when hasBias = 1. 16 x 16 tiles, one output per thread.
     */
    public static void gemm(KernelContext kc, FloatArray x, ByteArray xq, FloatArray sa, ByteArray w, FloatArray sw, FloatArray bias, FloatArray special, FloatArray y, int rows, int m, int k,
            int aFormat, int wFormat, int hasBias) {
        float nan = special.get(0);
        float inf = special.get(1);
        float[] as = kc.allocateFloatLocalArray(TILE * TILE);
        float[] bs = kc.allocateFloatLocalArray(TILE * TILE);
        int tx = kc.localIdx;
        int ty = kc.localIdy;
        int col = kc.groupIdx * TILE + tx;
        int row = kc.groupIdy * TILE + ty;
        int wRow = kc.groupIdx * TILE + ty;
        float scaleA = aFormat >= 0 ? sa.get(0) : 1.0f;
        float acc = 0.0f;
        for (int k0 = 0; k0 < k; k0 += TILE) {
            float av = 0.0f;
            if ((row < rows) & (k0 + tx < k)) {
                if (aFormat >= 0) {
                    av = decode(xq.get(row * k + k0 + tx), aFormat, nan, inf);
                } else {
                    av = x.get(row * k + k0 + tx);
                }
            }
            as[ty * TILE + tx] = av;
            float bv = 0.0f;
            if ((wRow < m) & (k0 + tx < k)) {
                bv = decode(w.get(wRow * k + k0 + tx), wFormat, nan, inf);
            }
            bs[ty * TILE + tx] = bv;
            kc.localBarrier();
            for (int i = 0; i < TILE; i++) {
                acc += as[ty * TILE + i] * bs[tx * TILE + i];
            }
            kc.localBarrier();
        }
        if ((row < rows) & (col < m)) {
            float v = acc * scaleA * sw.get(0);
            if (hasBias == 1) {
                v += bias.get(col);
            }
            y.set(row * m + col, v);
        }
    }
}
