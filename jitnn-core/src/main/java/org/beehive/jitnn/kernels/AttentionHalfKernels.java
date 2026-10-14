package org.beehive.jitnn.kernels;

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
