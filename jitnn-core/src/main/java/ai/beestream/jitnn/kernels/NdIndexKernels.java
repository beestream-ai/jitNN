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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * N-dimensional gather and scatter (ONNX GatherND and ScatterND, batch_dims 0): an index row of k
 * integers addresses a slice of the data, the trailing data axes after the first k.
 */
public final class NdIndexKernels {

    public static final int NONE = 0;
    public static final int ADD = 1;
    public static final int MUL = 2;

    private NdIndexKernels() {
    }

    /**
     * Flat data offset of each of the {@code rows} index rows [rows, k]: sum of idx * stride over the
     * first k data axes, negative indices counted from the end. meta = [k, dims[0..k), strides[0..k)].
     */
    public static void ndOffsets(KernelContext kc, IntArray indices, IntArray meta, IntArray offsets, int rows) {
        int r = kc.globalIdx;
        if (r < rows) {
            int k = meta.get(0);
            int off = 0;
            for (int t = 0; t < k; t++) {
                int v = indices.get(r * k + t);
                int size = meta.get(1 + t);
                if (v < 0) {
                    v += size;
                }
                off += v * meta.get(1 + k + t);
            }
            offsets.set(r, off);
        }
    }

    /** out[r, j] = data[offsets[r] + j] for slices of {@code slice} elements; one thread per output. */
    public static void gatherND(KernelContext kc, FloatArray data, IntArray offsets, FloatArray out, int slice, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int r = i / slice;
            int j = i - r * slice;
            out.set(i, data.get(offsets.get(r) + j));
        }
    }

    /**
     * out = data with each update slice r written to (NONE), added to (ADD) or multiplied into (MUL)
     * the slice at offsets[r], in row order (deterministic: a later duplicate wins for NONE). One
     * thread per output element, scanning the index rows.
     */
    public static void scatterND(KernelContext kc, FloatArray data, IntArray offsets, FloatArray updates, FloatArray out, int rows, int slice, int reduction, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int j = i % slice;
            int base = i - j;
            float acc = data.get(i);
            for (int r = 0; r < rows; r++) {
                if (offsets.get(r) == base) {
                    float u = updates.get(r * slice + j);
                    if (reduction == ADD) {
                        acc = acc + u;
                    } else if (reduction == MUL) {
                        acc = acc * u;
                    } else {
                        acc = u;
                    }
                }
            }
            out.set(i, acc);
        }
    }
}
