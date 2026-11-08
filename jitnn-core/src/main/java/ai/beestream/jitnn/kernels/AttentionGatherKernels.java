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
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * FP32 attention operands moved between a layout read through strides (packed q, k, v or
 * [B, T, H, d]) and contiguous [z, rows, d] per problem z = batch * heads + head, for a library's
 * fused attention. The strides are FlashAttention's meta array.
 */
public final class AttentionGatherKernels {

    private AttentionGatherKernels() {
    }

    /**
     * qz [z, t, d], kz and vz [z, s, d] from q, k, v in one pass; total = z max(t, s) d. With hasBias,
     * bias[index % rowLen] is added to every element read (the bias of the projection that produced a
     * packed q, k, v, of row length rowLen).
     */
    public static void gatherQkv(KernelContext kc, FloatArray q, FloatArray k, FloatArray v, FloatArray qz, FloatArray kz, FloatArray vz, IntArray meta, int t, int s, int d,
            int total, FloatArray bias, int rowLen, int hasBias) {
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
                int at = meta.get(5) + b * meta.get(6) + h * meta.get(7) + row * meta.get(8) + j;
                float x = q.get(at);
                if (hasBias != 0) {
                    x = x + bias.get(at % rowLen);
                }
                qz.set(i, x);
            }
            if (i < problems * s * d) {
                int row = (i / d) % s;
                int z = i / (d * s);
                int b = z / heads;
                int h = z - b * heads;
                int ak = meta.get(9) + b * meta.get(10) + h * meta.get(11) + row * meta.get(12) + j;
                int av = meta.get(13) + b * meta.get(14) + h * meta.get(15) + row * meta.get(16) + j;
                float xk = k.get(ak);
                float xv = v.get(av);
                if (hasBias != 0) {
                    xk = xk + bias.get(ak % rowLen);
                    xv = xv + bias.get(av % rowLen);
                }
                kz.set(i, xk);
                vz.set(i, xv);
            }
        }
    }

    /** {@link #gatherQkv} on FP16 tensors (an FP16 bias, FP32 addition). */
    public static void gatherQkvHalf(KernelContext kc, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, HalfFloatArray qz, HalfFloatArray kz, HalfFloatArray vz, IntArray meta,
            int t, int s, int d, int total, HalfFloatArray bias, int rowLen, int hasBias) {
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
                int at = meta.get(5) + b * meta.get(6) + h * meta.get(7) + row * meta.get(8) + j;
                float x = q.get(at).getFloat32();
                if (hasBias != 0) {
                    x = x + bias.get(at % rowLen).getFloat32();
                }
                qz.set(i, new HalfFloat(x));
            }
            if (i < problems * s * d) {
                int row = (i / d) % s;
                int z = i / (d * s);
                int b = z / heads;
                int h = z - b * heads;
                int ak = meta.get(9) + b * meta.get(10) + h * meta.get(11) + row * meta.get(12) + j;
                int av = meta.get(13) + b * meta.get(14) + h * meta.get(15) + row * meta.get(16) + j;
                float xk = k.get(ak).getFloat32();
                float xv = v.get(av).getFloat32();
                if (hasBias != 0) {
                    xk = xk + bias.get(ak % rowLen).getFloat32();
                    xv = xv + bias.get(av % rowLen).getFloat32();
                }
                kz.set(i, new HalfFloat(xk));
                vz.set(i, new HalfFloat(xv));
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
