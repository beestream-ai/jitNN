package ai.beestream.jitnn.kernels;

import ai.beestream.jitnn.DType;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Dequantisation of GGUF's block formats Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K and Q3_K,
 * in the byte layouts of llama.cpp's ggml (block_q4_0 ... block_q6_K). Each value is decoded on its own (one thread per
 * value), with byte fields unpacked by division and remainder on non-negative ints rather than
 * shifts and masks, which the CUDA backend treats as unsigned. The matrix products on these formats
 * are jitllm's kernels (see {@link ai.beestream.jitnn.ops.Gguf}).
 */
public final class GgufKernels {

    public static final int Q4_0 = 0;
    public static final int Q4_1 = 1;
    public static final int Q4_K = 2;
    public static final int Q5_K = 3;
    public static final int Q6_K = 4;
    public static final int Q5_0 = 5;
    public static final int Q5_1 = 6;
    public static final int Q2_K = 7;
    public static final int Q3_K = 8;

    private GgufKernels() {
    }

    public static int code(DType t) {
        return switch (t) {
            case Q4_0 -> Q4_0;
            case Q4_1 -> Q4_1;
            case Q4_K -> Q4_K;
            case Q5_K -> Q5_K;
            case Q6_K -> Q6_K;
            case Q5_0 -> Q5_0;
            case Q5_1 -> Q5_1;
            case Q2_K -> Q2_K;
            case Q3_K -> Q3_K;
            default -> throw new IllegalArgumentException("not a GGUF format with jitNN dequantisation: " + t);
        };
    }

    private static int u8(ByteArray w, int at) {
        int b = w.get(at);
        if (b < 0) {
            b = b + 256;
        }
        return b;
    }

    private static float f16(ByteArray w, int at) {
        return w.getHalfFloat(at).getFloat32();
    }

    /** Q4_K and Q5_K's 6-bit scale (part 0) or min (part 1) of sub-block j, from the 12 scale bytes at s. */
    private static int scaleMin(ByteArray w, int s, int j, int part) {
        int v;
        if (j < 4) {
            v = u8(w, s + j + 4 * part) % 64;
        } else {
            int low = part == 0 ? u8(w, s + j + 4) % 16 : u8(w, s + j + 4) / 16;
            v = low + (u8(w, s + j - 4 + 4 * part) / 64) * 16;
        }
        return v;
    }

    public static float q4_0(ByteArray w, int i) {
        int base = (i / 32) * 18;
        int e = i % 32;
        int b = u8(w, base + 2 + e % 16);
        int q = e < 16 ? b % 16 : b / 16;
        return (q - 8) * f16(w, base);
    }

    public static float q4_1(ByteArray w, int i) {
        int base = (i / 32) * 20;
        int e = i % 32;
        int b = u8(w, base + 4 + e % 16);
        int q = e < 16 ? b % 16 : b / 16;
        return q * f16(w, base) + f16(w, base + 2);
    }

    public static float q4_k(ByteArray w, int i) {
        int base = (i / 256) * 144;
        int e = i % 256;
        int j = e / 64;
        int half = (e % 64) / 32;
        int l = e % 32;
        int sub = 2 * j + half;
        int b = u8(w, base + 16 + 32 * j + l);
        int q = half == 0 ? b % 16 : b / 16;
        return f16(w, base) * scaleMin(w, base + 4, sub, 0) * q - f16(w, base + 2) * scaleMin(w, base + 4, sub, 1);
    }

    public static float q5_k(ByteArray w, int i) {
        int base = (i / 256) * 176;
        int e = i % 256;
        int j = e / 64;
        int half = (e % 64) / 32;
        int l = e % 32;
        int sub = 2 * j + half;
        int b = u8(w, base + 48 + 32 * j + l);
        int q = half == 0 ? b % 16 : b / 16;
        int hbit = (u8(w, base + 16 + l) / (1 << sub)) % 2;
        q = q + 16 * hbit;
        return f16(w, base) * scaleMin(w, base + 4, sub, 0) * q - f16(w, base + 2) * scaleMin(w, base + 4, sub, 1);
    }

    public static float q6_k(ByteArray w, int i) {
        int base = (i / 256) * 210;
        int e = i % 256;
        int n = e / 128;
        int r = e % 128;
        int k = r / 32;
        int l = r % 32;
        int lb = u8(w, base + 64 * n + l + (k % 2) * 32);
        int low = k < 2 ? lb % 16 : lb / 16;
        int hb = (u8(w, base + 128 + 32 * n + l) / (1 << (2 * k))) % 4;
        int q = low + 16 * hb - 32;
        int sc = w.get(base + 192 + 8 * n + l / 16 + 2 * k);
        return f16(w, base + 208) * sc * q;
    }

    /** Bit {@code bit} (0..31) of the little-endian 32-bit word at {@code at}. */
    private static int bit32(ByteArray w, int at, int bit) {
        return (u8(w, at + bit / 8) / (1 << (bit % 8))) % 2;
    }

    /** Q5_0: FP16 scale d, 32 high bits, 16 bytes of low nibbles; (q - 16) d. */
    public static float q5_0(ByteArray w, int i) {
        int base = (i / 32) * 22;
        int e = i % 32;
        int b = u8(w, base + 6 + e % 16);
        int q = (e < 16 ? b % 16 : b / 16) + 16 * bit32(w, base + 2, e);
        return (q - 16) * f16(w, base);
    }

    /** Q5_1: FP16 scale d and minimum m, 32 high bits, 16 bytes of low nibbles; q d + m. */
    public static float q5_1(ByteArray w, int i) {
        int base = (i / 32) * 24;
        int e = i % 32;
        int b = u8(w, base + 8 + e % 16);
        int q = (e < 16 ? b % 16 : b / 16) + 16 * bit32(w, base + 4, e);
        return q * f16(w, base) + f16(w, base + 2);
    }

    /** Q2_K: 16 scale bytes (4-bit scale, 4-bit minimum), 64 bytes of 2-bit values, FP16 d and dmin. */
    public static float q2_k(ByteArray w, int i) {
        int base = (i / 256) * 84;
        int e = i % 256;
        int n = e / 128;
        int j = (e % 128) / 32;
        int l = e % 32;
        int sc = u8(w, base + n * 8 + j * 2 + l / 16);
        int q = (u8(w, base + 16 + n * 32 + l) / (1 << (2 * j))) % 4;
        return f16(w, base + 80) * (sc % 16) * q - f16(w, base + 82) * (sc / 16);
    }

    /**
     * Q3_K's 6-bit scale {@code is} (0..15) from the 12 packed bytes at s: the low 4 bits from bytes
     * 0..7, the high 2 bits from bytes 8..11 (ggml's kmask unpacking).
     */
    private static int scale3(ByteArray w, int s, int is) {
        int low = is < 8 ? u8(w, s + is) % 16 : u8(w, s + is - 8) / 16;
        int high = (u8(w, s + 8 + is % 4) / (1 << (2 * (is / 4)))) % 4;
        return low + 16 * high;
    }

    /** Q3_K: 32 bytes of high bits, 64 bytes of 2-bit values, 12 bytes of packed 6-bit scales, FP16 d. */
    public static float q3_k(ByteArray w, int i) {
        int base = (i / 256) * 110;
        int e = i % 256;
        int n = e / 128;
        int j = (e % 128) / 32;
        int l = e % 32;
        int is = n * 8 + j * 2 + l / 16;
        int q = (u8(w, base + 32 + n * 32 + l) / (1 << (2 * j))) % 4;
        int high = (u8(w, base + l) / (1 << (4 * n + j))) % 2;
        return f16(w, base + 108) * (scale3(w, base + 96, is) - 32) * (q - (high == 1 ? 0 : 4));
    }

    /** Value i of the tensor w in format {@code code} (host and device). */
    public static float value(ByteArray w, int code, int i) {
        float v;
        if (code == Q4_0) {
            v = q4_0(w, i);
        } else if (code == Q4_1) {
            v = q4_1(w, i);
        } else if (code == Q4_K) {
            v = q4_k(w, i);
        } else if (code == Q5_K) {
            v = q5_k(w, i);
        } else if (code == Q6_K) {
            v = q6_k(w, i);
        } else if (code == Q5_0) {
            v = q5_0(w, i);
        } else if (code == Q5_1) {
            v = q5_1(w, i);
        } else if (code == Q2_K) {
            v = q2_k(w, i);
        } else {
            v = q3_k(w, i);
        }
        return v;
    }

    public static void dequantizeQ4_0(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q4_0(w, i));
        }
    }

    public static void dequantizeQ4_1(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q4_1(w, i));
        }
    }

    public static void dequantizeQ4_K(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q4_k(w, i));
        }
    }

    public static void dequantizeQ5_K(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q5_k(w, i));
        }
    }

    public static void dequantizeQ6_K(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q6_k(w, i));
        }
    }

    public static void dequantizeQ5_0(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q5_0(w, i));
        }
    }

    public static void dequantizeQ5_1(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q5_1(w, i));
        }
    }

    public static void dequantizeQ2_K(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q2_k(w, i));
        }
    }

    public static void dequantizeQ3_K(KernelContext kc, ByteArray w, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, q3_k(w, i));
        }
    }
}
