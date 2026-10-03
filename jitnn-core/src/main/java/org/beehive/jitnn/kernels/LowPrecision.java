package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
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
}
