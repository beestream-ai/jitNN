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
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Attention in half precision on the tensor cores, unfused: q, k and v gathered per problem z (batch
 * x head) into contiguous FP16 matrices (k transposed), S = q k^T and O = P v as batched FP16 GEMMs,
 * the softmax in between one warp per row, and O merged back into the output's layout.
 */
public final class AttentionHalfKernels {

    private AttentionHalfKernels() {
    }

    /**
     * dst [z, rows, d] = the rows of operand src for problem z = batch * heads + head, element (z,
     * row, j) read at off + batch * sb + head * sh + row * st + j.
     */
    public static void gather(KernelContext kc, HalfFloatArray src, HalfFloatArray dst, int heads, int rows, int d, int off, int sb, int sh, int st, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int j = i % d;
            int row = (i / d) % rows;
            int z = i / (d * rows);
            int b = z / heads;
            int h = z - b * heads;
            dst.set(i, src.get(off + b * sb + h * sh + row * st + j));
        }
    }


    /**
     * {@link #gather} of q, k and v in one launch: element i of qz (if i &lt; t d z) and of kz and vz (if
     * i &lt; s d z), with the (offset, batch, head, token) strides of q, k and v at meta[5..16] and the
     * heads at meta[0] (FlashAttention's metadata).
     */
    public static void gatherQkv(KernelContext kc, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, HalfFloatArray qz, HalfFloatArray kz, HalfFloatArray vz,
            uk.ac.manchester.tornado.api.types.arrays.IntArray meta, int t, int s, int d, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int heads = meta.get(0);
            int j = i % d;
            if (i < qzSize(t, d, total, s)) {
                int row = (i / d) % t;
                int z = i / (d * t);
                int b = z / heads;
                int h = z - b * heads;
                qz.set(i, q.get(meta.get(5) + b * meta.get(6) + h * meta.get(7) + row * meta.get(8) + j));
            }
            int kRows = s;
            int z = i / (d * kRows);
            int row = (i / d) % kRows;
            int b = z / heads;
            int h = z - b * heads;
            if (i < kzSize(t, d, total, s)) {
                kz.set(i, k.get(meta.get(9) + b * meta.get(10) + h * meta.get(11) + row * meta.get(12) + j));
                vz.set(i, v.get(meta.get(13) + b * meta.get(14) + h * meta.get(15) + row * meta.get(16) + j));
            }
        }
    }

    /** The q elements (z t d) when total = z max(t, s) d. */
    private static int qzSize(int t, int d, int total, int s) {
        int big = t > s ? t : s;
        return total / big * t;
    }

    /** The k (and v) elements (z s d) when total = z max(t, s) d. */
    private static int kzSize(int t, int d, int total, int s) {
        int big = t > s ? t : s;
        return total / big * s;
    }

    /** As {@link #gather}, transposed: dst [z, d, rows] (k^T for a GEMM without a transposed operand). */
    public static void gatherTransposed(KernelContext kc, HalfFloatArray src, HalfFloatArray dst, int heads, int rows, int d, int off, int sb, int sh, int st, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int row = i % rows;
            int j = (i / rows) % d;
            int z = i / (d * rows);
            int b = z / heads;
            int h = z - b * heads;
            dst.set(i, src.get(off + b * sb + h * sh + row * st + j));
        }
    }

    /**
     * p = softmax(scale * s) per row of the FP16 scores s [rows, cols], written as FP16, one warp per
     * row (8 rows per 256-thread group); with {@code causal}, query t = row % queryLen sees keys 0 .. t.
     */
    public static void softmaxHalfWarp(KernelContext kc, HalfFloatArray s, HalfFloatArray p, int rows, int cols, float scale, int causal, int queryLen) {
        int lane = kc.localIdx % 32;
        int row = kc.groupIdx * 8 + kc.localIdx / 32;
        if (row < rows) {
            int base = row * cols;
            int valid = cols;
            if (causal != 0) {
                int limit = row % queryLen + 1;
                valid = limit < cols ? limit : cols;
            }
            float m = -3.0e38f;
            for (int c = lane; c < valid; c += 32) {
                float v = scale * s.get(base + c).getFloat32();
                m = v > m ? v : m;
            }
            for (int off = 16; off > 0; off = off / 2) {
                float o = kc.simdShuffleDown(m, off);
                m = o > m ? o : m;
            }
            m = kc.simdBroadcastFirst(m);
            float sum = 0.0f;
            for (int c = lane; c < valid; c += 32) {
                sum += TornadoMath.exp(scale * s.get(base + c).getFloat32() - m);
            }
            float inv = 1.0f / kc.simdSum(sum);
            for (int c = lane; c < cols; c += 32) {
                float v = 0.0f;
                if (c < valid) {
                    v = TornadoMath.exp(scale * s.get(base + c).getFloat32() - m) * inv;
                }
                p.set(base + c, new HalfFloat(v));
            }
        }
    }

    /** O [z, rows, d] (FP16) written into o in the output's layout, as {@link #gather} reads it. */
    public static void mergeHalf(KernelContext kc, HalfFloatArray src, HalfFloatArray o, int heads, int rows, int d, int off, int sb, int sh, int st, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int j = i % d;
            int row = (i / d) % rows;
            int z = i / (d * rows);
            int b = z / heads;
            int h = z - b * heads;
            o.set(off + b * sb + h * sh + row * st + j, src.get(i));
        }
    }

}
