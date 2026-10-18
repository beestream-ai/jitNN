package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * FP32 attention operands moved between a layout read through strides (packed q, k, v or
 * [B, T, H, d]) and contiguous [z, rows, d] per problem z = batch * heads + head, for a library's
 * fused attention. The strides are FlashAttention's meta array.
 */
public final class AttentionGatherKernels {

    private AttentionGatherKernels() {
    }

    /** qz [z, t, d], kz and vz [z, s, d] from q, k, v in one pass; total = z max(t, s) d. */
    public static void gatherQkv(KernelContext kc, FloatArray q, FloatArray k, FloatArray v, FloatArray qz, FloatArray kz, FloatArray vz, IntArray meta, int t, int s, int d,
            int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int heads = meta.get(0);
            int j = i % d;
            int big = t > s ? t : s;
            int problems = total / (big * d);
            if (i < problems * t * d) {
                int row = (i / d) % t;
                int z = i / (d * t);
                int b = z / heads;
                int h = z - b * heads;
                qz.set(i, q.get(meta.get(5) + b * meta.get(6) + h * meta.get(7) + row * meta.get(8) + j));
            }
            if (i < problems * s * d) {
                int row = (i / d) % s;
                int z = i / (d * s);
                int b = z / heads;
                int h = z - b * heads;
                kz.set(i, k.get(meta.get(9) + b * meta.get(10) + h * meta.get(11) + row * meta.get(12) + j));
                vz.set(i, v.get(meta.get(13) + b * meta.get(14) + h * meta.get(15) + row * meta.get(16) + j));
            }
        }
    }

    /** o in its layout from oz [z, rows, d]: element (z, row, j) written at off + batch sb + head sh + row st + j. */
    public static void merge(KernelContext kc, FloatArray oz, FloatArray o, int heads, int rows, int d, int off, int sb, int sh, int st, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int j = i % d;
            int row = (i / d) % rows;
            int z = i / (d * rows);
            int b = z / heads;
            int h = z - b * heads;
            o.set(off + b * sb + h * sh + row * st + j, oz.get(i));
        }
    }
}
