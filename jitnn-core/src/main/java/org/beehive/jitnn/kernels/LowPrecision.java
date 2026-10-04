package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

/**
 * Half-precision conversions. FP16 uses TornadoVM's HalfFloat (round to nearest even); bfloat16 is
 * stored as raw bits in a ShortArray and converted with bit operations (needs TornadoVM#1141 for
 * float-bit reinterpretation on OpenCL and Metal).
 */
public final class LowPrecision {

    private LowPrecision() {
    }

    /** FP32 to bfloat16 bits, rounding to nearest even; NaN stays a (quiet) NaN. */
    public static short toBF16(float v) {
        int bits = Float.floatToRawIntBits(v);
        int r;
        if (((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)) {
            r = (bits >> 16) | 0x40;
        } else {
            r = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
        }
        return (short) r;
    }

    /** bfloat16 bits to FP32 (exact). */
    public static float fromBF16(short b) {
        return Float.intBitsToFloat(((int) b) << 16);
    }

    /** 1 when v is NaN or infinite, tested on the bit pattern (comparisons with NaN are not reliable on every backend). */
    public static int nonFinite(float v) {
        return (Float.floatToRawIntBits(v) & 0x7F800000) == 0x7F800000 ? 1 : 0;
    }

    public static void f32ToF16(KernelContext kc, FloatArray x, HalfFloatArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, new HalfFloat(x.get(i)));
        }
    }

    public static void f16ToF32(KernelContext kc, HalfFloatArray x, FloatArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, x.get(i).getFloat32());
        }
    }

    public static void f32ToBF16(KernelContext kc, FloatArray x, ShortArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, toBF16(x.get(i)));
        }
    }

    public static void bf16ToF32(KernelContext kc, ShortArray x, FloatArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, fromBF16(x.get(i)));
        }
    }

    private static final int GROUP = 256;

    /**
     * Gradient statistics, one tensor per launch: partial[2 slot] = sum of g^2, partial[2 slot + 1]
     * = 1 when g holds a NaN or infinity. One work-group of 256 threads.
     */
    public static void sumSquaresFinite(KernelContext kc, FloatArray g, FloatArray partial, int slot, int n) {
        float[] sq = kc.allocateFloatLocalArray(GROUP);
        int[] bad = kc.allocateIntLocalArray(GROUP);
        int l = kc.localIdx;
        float s = 0.0f;
        int b = 0;
        for (int i = l; i < n; i += GROUP) {
            float v = g.get(i);
            s += v * v;
            b = b | nonFinite(v);
        }
        sq[l] = s;
        bad[l] = b;
        kc.localBarrier();
        for (int w = GROUP / 2; w > 0; w = w / 2) {
            if (l < w) {
                sq[l] = sq[l] + sq[l + w];
                bad[l] = bad[l] | bad[l + w];
            }
            kc.localBarrier();
        }
        if (l == 0) {
            partial.set(2 * slot, sq[0]);
            partial.set(2 * slot + 1, bad[0]);
        }
    }

    /**
     * stats = [norm, factor, skip] from the partials of {@code count} tensors: norm = preScale *
     * sqrt(sum), factor = preScale * min(1, maxNorm / (norm + 1e-6)) when clipping (else
     * preScale), skip = 1 when any value was NaN or infinite. One thread.
     */
    public static void gradStats(KernelContext kc, FloatArray partial, FloatArray preScale, FloatArray stats, int count, float maxNorm, int clip) {
        if (kc.globalIdx == 0) {
            float total = 0.0f;
            float bad = 0.0f;
            for (int i = 0; i < count; i++) {
                total += partial.get(2 * i);
                bad = partial.get(2 * i + 1) > bad ? partial.get(2 * i + 1) : bad;
            }
            float scale = preScale.get(0);
            float norm = TornadoMath.sqrt(total) * scale;
            float factor = scale;
            if (clip != 0) {
                float c = maxNorm / (norm + 1e-6f);
                factor = c < 1.0f ? scale * c : scale;
            }
            stats.set(0, norm);
            stats.set(1, factor);
            stats.set(2, bad);
        }
    }
}
