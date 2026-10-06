package org.beehive.jitnn.kernels;

import org.beehive.jitnn.DType;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Dequantisation of GGUF's block formats Q4_0, Q4_1, Q4_K, Q5_K and Q6_K, in the byte layouts of
 * llama.cpp's ggml (block_q4_0 ... block_q6_K). Each value is decoded on its own (one thread per
 * value), with byte fields unpacked by division and remainder on non-negative ints rather than
 * shifts and masks, which the CUDA backend treats as unsigned. The matrix products on these formats
 * are jitllm's kernels (see {@link org.beehive.jitnn.ops.Gguf}).
 */
public final class GgufKernels {

    public static final int Q4_0 = 0;
    public static final int Q4_1 = 1;
    public static final int Q4_K = 2;
    public static final int Q5_K = 3;
    public static final int Q6_K = 4;

    private GgufKernels() {
    }

    public static int code(DType t) {
        return switch (t) {
            case Q4_0 -> Q4_0;
            case Q4_1 -> Q4_1;
            case Q4_K -> Q4_K;
            case Q5_K -> Q5_K;
            case Q6_K -> Q6_K;
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
        } else {
            v = q6_k(w, i);
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
}
