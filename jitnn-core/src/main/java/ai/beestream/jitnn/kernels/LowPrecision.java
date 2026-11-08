/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.kernels;

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

    /** Truncates toward zero, as a Java (int) cast. */
    public static void f32ToI32(KernelContext kc, FloatArray x, IntArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, (int) x.get(i));
        }
    }

    public static void i32ToF32(KernelContext kc, IntArray x, FloatArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, (float) x.get(i));
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


    /** Threads of the one work-group that runs {@link #gradStats}. */
    public static final int STATS_GROUP = 256;

    /**
     * stats = [norm, factor, skip] from {@code count} partials: norm = preScale * sqrt(sum),
     * factor = preScale * min(1, maxNorm / (norm + 1e-6)) when clipping (else preScale), skip = 1
     * when any value was NaN or infinite. One work-group of {@link #STATS_GROUP} threads.
     */
    public static void gradStats(KernelContext kc, FloatArray partial, FloatArray preScale, FloatArray stats, int count, float maxNorm, int clip) {
        float[] sq = kc.allocateFloatLocalArray(STATS_GROUP);
        float[] bd = kc.allocateFloatLocalArray(STATS_GROUP);
        int l = kc.localIdx;
        float total = 0.0f;
        float bad = 0.0f;
        for (int i = l; i < count; i += STATS_GROUP) {
            total += partial.get(2 * i);
            float b = partial.get(2 * i + 1);
            bad = b > bad ? b : bad;
        }
        sq[l] = total;
        bd[l] = bad;
        kc.localBarrier();
        for (int r = STATS_GROUP / 2; r > 0; r = r / 2) {
            if (l < r) {
                sq[l] = sq[l] + sq[l + r];
                bd[l] = bd[l + r] > bd[l] ? bd[l + r] : bd[l];
            }
            kc.localBarrier();
        }
        if (l == 0) {
            float scale = preScale.get(0);
            float norm = TornadoMath.sqrt(sq[0]) * scale;
            float factor = scale;
            if (clip != 0) {
                float c = maxNorm / (norm + 1e-6f);
                factor = c < 1.0f ? scale * c : scale;
            }
            stats.set(0, norm);
            stats.set(1, factor);
            stats.set(2, bd[0]);
        }
    }
}
