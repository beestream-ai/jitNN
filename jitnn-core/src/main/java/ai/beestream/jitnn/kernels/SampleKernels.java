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

/** Top-k selection and multinomial sampling, one work-group of 256 threads per row. */
public final class SampleKernels {

    private static final int GROUP = 256;

    private SampleKernels() {
    }

    /**
     * The k largest entries of each row, in descending order (ties: lower index first). Pass q
     * picks the best entry ordered after the one picked by pass q - 1, with a local reduction.
     */
    public static void topkRows(KernelContext kc, FloatArray x, FloatArray values, IntArray indices, int cols, int k) {
        float[] bv = kc.allocateFloatLocalArray(GROUP);
        int[] bi = kc.allocateIntLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        float prevV = Float.POSITIVE_INFINITY;
        int prevI = -1;
        for (int q = 0; q < k; q++) {
            float best = Float.NEGATIVE_INFINITY;
            int bestI = cols;
            for (int c = l; c < cols; c += GROUP) {
                float v = x.get(base + c);
                boolean after = (v < prevV) | ((v == prevV) & (c > prevI));
                boolean better = (v > best) | ((v == best) & (c < bestI));
                if (after & better) {
                    best = v;
                    bestI = c;
                }
            }
            bv[l] = best;
            bi[l] = bestI;
            kc.localBarrier();
            for (int s = GROUP / 2; s > 0; s = s / 2) {
                if (l < s) {
                    float ov = bv[l + s];
                    int oi = bi[l + s];
                    if ((ov > bv[l]) | ((ov == bv[l]) & (oi < bi[l]))) {
                        bv[l] = ov;
                        bi[l] = oi;
                    }
                }
                kc.localBarrier();
            }
            prevV = bv[0];
            prevI = bi[0];
            if (l == 0) {
                values.set(row * k + q, prevV);
                indices.set(row * k + q, prevI);
            }
            kc.localBarrier();
        }
    }

    /** dx[r, c] = sum of dy[r, q] over the picks q with indices[r, q] == c; n = rows * cols. */
    public static void topkBackward(KernelContext kc, FloatArray dy, IntArray indices, FloatArray dx, int cols, int k, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int row = i / cols;
            int c = i - row * cols;
            float acc = 0.0f;
            for (int q = 0; q < k; q++) {
                if (indices.get(row * k + q) == c) {
                    acc += dy.get(row * k + q);
                }
            }
            dx.set(i, acc);
        }
    }

    /**
     * One draw per row from the unnormalised non-negative weights of that row: u = uniform *
     * total, from Philox keyed by seed[0..1] with counter (row, stream); the result is the first
     * index whose running sum exceeds u. Each thread sums a contiguous chunk; thread 0 walks the
     * chunk sums and then the chosen chunk.
     */
    public static void multinomialRows(KernelContext kc, FloatArray weights, IntArray seed, IntArray out, int cols, int stream, int rounds) {
        float[] sums = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        int chunk = (cols + GROUP - 1) / GROUP;
        int c0 = l * chunk;
        int c1 = c0 + chunk < cols ? c0 + chunk : cols;
        float s = 0.0f;
        for (int c = c0; c < c1; c++) {
            s += weights.get(base + c);
        }
        sums[l] = s;
        kc.localBarrier();
        if (l == 0) {
            float total = 0.0f;
            for (int j = 0; j < GROUP; j++) {
                total += sums[j];
            }
            float u = RandomKernels.toUnit(RandomKernels.philoxWord(row, stream, seed.get(0), seed.get(1), 0, rounds)) * total;
            float acc = 0.0f;
            int pick = -1;
            int last = 0;
            for (int j = 0; j < GROUP; j++) {
                int start = j * chunk;
                int end = start + chunk < cols ? start + chunk : cols;
                if ((pick < 0) & (start < cols)) {
                    if (acc + sums[j] > u) {
                        for (int c = start; c < end; c++) {
                            float w = weights.get(base + c);
                            if ((pick < 0) & (w > 0.0f)) {
                                acc += w;
                                if (acc > u) {
                                    pick = c;
                                }
                            }
                        }
                    } else {
                        acc += sums[j];
                    }
                }
            }
            if (pick < 0) {
                // u reached the total through rounding: take the last entry with weight
                for (int c = 0; c < cols; c++) {
                    if (weights.get(base + c) > 0.0f) {
                        last = c;
                    }
                }
                pick = last;
            }
            out.set(row, pick);
        }
    }
}
