package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.FP8;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * FP8 (OCP E4M3 and E5M2) as a storage type with a per-tensor scale: value = decode(byte) * scale.
 * Encoding and decoding use TornadoVM's {@link FP8} conversions, written with integer and float
 * arithmetic only, so they run on every backend; the GEMMs decode their FP8 operands while
 * loading tiles into local memory and accumulate in FP32.
 */
public final class Fp8Kernels {

    public static final int E4M3 = 0;
    public static final int E5M2 = 1;
    public static final int TILE = 16;

    private Fp8Kernels() {
    }

    public static float decode(byte b, int format) {
        float v;
        if (format == E4M3) {
            v = FP8.e4m3ToFloat(b);
        } else {
            v = FP8.e5m2ToFloat(b);
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

    public static void dequantize(KernelContext kc, ByteArray q, FloatArray scale, FloatArray y, int format, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, decode(q.get(i), format) * scale.get(0));
        }
    }

    /**
     * y [rows, m] = x [rows, k] (FP32, or FP8 when aFormat >= 0, scaled by sa) times w^T, w [m, k]
     * FP8 scaled by sw, plus bias when hasBias = 1. 16 x 16 tiles, one output per thread.
     */
    public static void gemm(KernelContext kc, FloatArray x, ByteArray xq, FloatArray sa, ByteArray w, FloatArray sw, FloatArray bias, FloatArray y, int rows, int m, int k,
            int aFormat, int wFormat, int hasBias) {
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
                    av = decode(xq.get(row * k + k0 + tx), aFormat);
                } else {
                    av = x.get(row * k + k0 + tx);
                }
            }
            as[ty * TILE + tx] = av;
            float bv = 0.0f;
            if ((wRow < m) & (k0 + tx < k)) {
                bv = decode(w.get(wRow * k + k0 + tx), wFormat);
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
