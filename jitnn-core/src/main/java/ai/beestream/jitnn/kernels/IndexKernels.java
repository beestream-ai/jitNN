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
 * Index-based selection along one axis of a tensor seen as [outer, n, inner] (index tensors as
 * [outer, m, inner] for gather and scatter), and segment reductions over row ranges. Gradients and
 * scatters gather per destination element, so they are deterministic and use no atomics.
 */
public final class IndexKernels {

    public static final int SUM = 0;
    public static final int MEAN = 1;
    public static final int MAX = 2;

    private IndexKernels() {
    }

    /** out[o, j, q] = x[o, idx[j], q] for j &lt; m (index_select). */
    public static void indexSelect(KernelContext kc, FloatArray x, IntArray idx, FloatArray out, int n, int m, int inner, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int q = i % inner;
            int j = (i / inner) % m;
            int o = i / (inner * m);
            out.set(i, x.get((o * n + idx.get(j)) * inner + q));
        }
    }

    /** Gradient of index_select: dx[o, r, q] = sum of dy[o, j, q] over j with idx[j] == r. */
    public static void indexSelectBackward(KernelContext kc, FloatArray dy, IntArray idx, FloatArray dx, int n, int m, int inner, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int q = i % inner;
            int r = (i / inner) % n;
            int o = i / (inner * n);
            float acc = 0.0f;
            for (int j = 0; j < m; j++) {
                if (idx.get(j) == r) {
                    acc += dy.get((o * m + j) * inner + q);
                }
            }
            dx.set(i, acc);
        }
    }

    /** out[o, j, q] = x[o, index[o, j, q], q] (torch.gather). */
    public static void gather(KernelContext kc, FloatArray x, IntArray index, FloatArray out, int n, int m, int inner, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int q = i % inner;
            int o = i / (inner * m);
            out.set(i, x.get((o * n + index.get(i)) * inner + q));
        }
    }

    /**
     * Scatter-add into a copy of {@code base}: out[o, r, q] = base[o, r, q] + sum of src[o, j, q]
     * over j with index[o, j, q] == r (with {@code add} 0, the last such src replaces base instead).
     */
    public static void scatter(KernelContext kc, FloatArray base, IntArray index, FloatArray src, FloatArray out, int n, int m, int inner, int add, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int q = i % inner;
            int r = (i / inner) % n;
            int o = i / (inner * n);
            float acc = base.get(i);
            for (int j = 0; j < m; j++) {
                int at = (o * m + j) * inner + q;
                if (index.get(at) == r) {
                    acc = add != 0 ? acc + src.get(at) : src.get(at);
                }
            }
            out.set(i, acc);
        }
    }

    /**
     * Reduction of each row segment: out[b, d] over rows offsets[b] .. ends[b] - 1 of x [rows, D],
     * sum, mean (empty: 0) or max (also writing the winning row to arg; empty: 0 and -1).
     */
    public static void segmentReduce(KernelContext kc, FloatArray x, IntArray starts, IntArray ends, FloatArray out, IntArray arg, int dim, int mode, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int b = i / dim;
            int d = i - b * dim;
            int s = starts.get(b);
            int e = ends.get(b);
            float acc = 0.0f;
            float best = -3.4028235e38f;
            int at = -1;
            for (int r = s; r < e; r++) {
                float v = x.get(r * dim + d);
                acc += v;
                if (v > best) {
                    best = v;
                    at = r;
                }
            }
            if (mode == MAX) {
                out.set(i, at < 0 ? 0.0f : best);
                arg.set(i, at);
            } else if (mode == MEAN) {
                out.set(i, e > s ? acc / (e - s) : 0.0f);
            } else {
                out.set(i, acc);
            }
        }
    }

    /** Gradient of {@link #segmentReduce}: dx[r, d] from the segment holding row r (rowSeg[r], or -1). */
    public static void segmentReduceBackward(KernelContext kc, FloatArray dy, IntArray rowSeg, IntArray starts, IntArray ends, IntArray arg, FloatArray dx, int dim, int mode, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int r = i / dim;
            int d = i - r * dim;
            int b = rowSeg.get(r);
            float v = 0.0f;
            if (b >= 0) {
                float g = dy.get(b * dim + d);
                if (mode == MAX) {
                    v = arg.get(b * dim + d) == r ? g : 0.0f;
                } else if (mode == MEAN) {
                    v = g / (ends.get(b) - starts.get(b));
                } else {
                    v = g;
                }
            }
            dx.set(i, v);
        }
    }
}
