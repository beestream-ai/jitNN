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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Dense decompositions of batches of small matrices, row-major [B, rows, cols]. The factorising
 * kernels run one work-group per matrix and work in global memory, with a work-group barrier
 * (global fence) between dependent steps; the triangular solve runs one thread per right-hand
 * side column. Conventions follow LAPACK, so the results match PyTorch's:
 * <ul>
 * <li>LU: partial pivoting on the largest magnitude (first on ties); L unit lower and U packed in
 * one matrix; perm[i] is the row of A that ends up in row i, sign the permutation's parity.
 * <li>QR: Householder reflectors with R's diagonal -sign(x0) |x| (dlarfg); reduced Q.
 * <li>eigh, SVD: cyclic Jacobi with a round-robin pair order (n/2 disjoint rotations per round);
 * eigenvalues ascending, singular values descending, and every vector pair signed so that the
 * largest-magnitude component of the chosen vector is positive.
 * </ul>
 */
public final class LinalgKernels {

    private LinalgKernels() {
    }

    /** LU factorisation with partial pivoting; pivot[b] is scratch. */
    public static void lu(KernelContext kc, FloatArray a, FloatArray lu, IntArray perm, FloatArray sign, IntArray pivot, int n, int group) {
        int b = kc.groupIdx;
        int tid = kc.localIdx;
        int base = b * n * n;
        for (int idx = tid; idx < n * n; idx += group) {
            lu.set(base + idx, a.get(base + idx));
        }
        for (int i = tid; i < n; i += group) {
            perm.set(b * n + i, i);
        }
        if (tid == 0) {
            sign.set(b, 1.0f);
        }
        kc.globalBarrier();
        for (int k = 0; k < n; k++) {
            if (tid == 0) {
                int p = k;
                float best = TornadoMath.abs(lu.get(base + k * n + k));
                for (int i = k + 1; i < n; i++) {
                    float v = TornadoMath.abs(lu.get(base + i * n + k));
                    if (v > best) {
                        best = v;
                        p = i;
                    }
                }
                pivot.set(b, p);
                if (p != k) {
                    int t = perm.get(b * n + k);
                    perm.set(b * n + k, perm.get(b * n + p));
                    perm.set(b * n + p, t);
                    sign.set(b, -sign.get(b));
                }
            }
            kc.globalBarrier();
            int p = pivot.get(b);
            if (p != k) {
                for (int j = tid; j < n; j += group) {
                    float t = lu.get(base + k * n + j);
                    lu.set(base + k * n + j, lu.get(base + p * n + j));
                    lu.set(base + p * n + j, t);
                }
            }
            kc.globalBarrier();
            float d = lu.get(base + k * n + k);
            if (d != 0.0f) {
                for (int i = k + 1 + tid; i < n; i += group) {
                    lu.set(base + i * n + k, lu.get(base + i * n + k) / d);
                }
            }
            kc.globalBarrier();
            int w = n - k - 1;
            for (int idx = tid; idx < w * w; idx += group) {
                int i = k + 1 + idx / w;
                int j = k + 1 + idx % w;
                lu.set(base + i * n + j, lu.get(base + i * n + j) - lu.get(base + i * n + k) * lu.get(base + k * n + j));
            }
            kc.globalBarrier();
        }
    }

    /**
     * Solves op(T) X = R for X [B, n, k], one thread per (matrix, column); op(T) is T or (trans = 1)
     * its transpose, lower or (upper = 1) upper triangular, with a unit diagonal when unit = 1 (the
     * stored diagonal and the other triangle are never read).
     */
    public static void triSolve(KernelContext kc, FloatArray t, FloatArray r, FloatArray x, int n, int k, int total, int upper, int unit, int trans) {
        int id = kc.globalIdx;
        if (id < total) {
            int b = id / k;
            int c = id % k;
            int tb = b * n * n;
            int xb = b * n * k;
            int rs = trans == 1 ? 1 : n;
            int cs = trans == 1 ? n : 1;
            for (int step = 0; step < n; step++) {
                int i = upper == 1 ? n - 1 - step : step;
                float s = r.get(xb + i * k + c);
                if (upper == 1) {
                    for (int j = i + 1; j < n; j++) {
                        s -= t.get(tb + i * rs + j * cs) * x.get(xb + j * k + c);
                    }
                } else {
                    for (int j = 0; j < i; j++) {
                        s -= t.get(tb + i * rs + j * cs) * x.get(xb + j * k + c);
                    }
                }
                if (unit == 0) {
                    s = s / t.get(tb + i * rs + i * cs);
                }
                x.set(xb + i * k + c, s);
            }
        }
    }

    /** Cholesky factor L (lower, zero above the diagonal) of the lower triangle of a. */
    public static void cholesky(KernelContext kc, FloatArray a, FloatArray l, int n, int group) {
        int b = kc.groupIdx;
        int tid = kc.localIdx;
        int base = b * n * n;
        for (int idx = tid; idx < n * n; idx += group) {
            int i = idx / n;
            int j = idx % n;
            l.set(base + idx, j <= i ? a.get(base + idx) : 0.0f);
        }
        kc.globalBarrier();
        for (int k = 0; k < n; k++) {
            if (tid == 0) {
                l.set(base + k * n + k, TornadoMath.sqrt(l.get(base + k * n + k)));
            }
            kc.globalBarrier();
            float d = l.get(base + k * n + k);
            for (int i = k + 1 + tid; i < n; i += group) {
                l.set(base + i * n + k, l.get(base + i * n + k) / d);
            }
            kc.globalBarrier();
            int w = n - k - 1;
            for (int idx = tid; idx < w * w; idx += group) {
                int i = k + 1 + idx / w;
                int j = k + 1 + idx % w;
                if (j <= i) {
                    l.set(base + i * n + j, l.get(base + i * n + j) - l.get(base + i * n + k) * l.get(base + j * n + k));
                }
            }
            kc.globalBarrier();
        }
    }

    /**
     * Reduced Householder QR of a [B, m, n], m >= n, into out [B, m + n, n]: Q in rows 0..m-1, R in
     * rows m..m+n-1. r and v [B, m, n] are scratch (the working matrix and the unit reflectors).
     */
    public static void qr(KernelContext kc, FloatArray a, FloatArray out, FloatArray r, FloatArray v, int m, int n, int group) {
        int b = kc.groupIdx;
        int tid = kc.localIdx;
        int base = b * m * n;
        int ob = b * (m + n) * n;
        for (int idx = tid; idx < m * n; idx += group) {
            r.set(base + idx, a.get(base + idx));
        }
        kc.globalBarrier();
        for (int k = 0; k < n; k++) {
            if (tid == 0) {
                float x0 = r.get(base + k * n + k);
                float sig = 0.0f;
                for (int i = k + 1; i < m; i++) {
                    float x = r.get(base + i * n + k);
                    sig += x * x;
                }
                if (sig == 0.0f) {
                    for (int i = k; i < m; i++) {
                        v.set(base + i * n + k, 0.0f);
                    }
                } else {
                    float norm = TornadoMath.sqrt(x0 * x0 + sig);
                    float alpha = x0 >= 0.0f ? -norm : norm;
                    float vk = x0 - alpha;
                    float vn = TornadoMath.sqrt(vk * vk + sig);
                    v.set(base + k * n + k, vk / vn);
                    for (int i = k + 1; i < m; i++) {
                        v.set(base + i * n + k, r.get(base + i * n + k) / vn);
                    }
                }
            }
            kc.globalBarrier();
            for (int j = k + tid; j < n; j += group) {
                float dot = 0.0f;
                for (int i = k; i < m; i++) {
                    dot += v.get(base + i * n + k) * r.get(base + i * n + j);
                }
                for (int i = k; i < m; i++) {
                    r.set(base + i * n + j, r.get(base + i * n + j) - 2.0f * v.get(base + i * n + k) * dot);
                }
            }
            kc.globalBarrier();
        }
        for (int idx = tid; idx < m * n; idx += group) {
            int i = idx / n;
            int j = idx % n;
            out.set(ob + idx, i == j ? 1.0f : 0.0f);
        }
        for (int idx = tid; idx < n * n; idx += group) {
            int i = idx / n;
            int j = idx % n;
            out.set(ob + m * n + idx, j >= i ? r.get(base + idx) : 0.0f);
        }
        kc.globalBarrier();
        for (int k = n - 1; k >= 0; k--) {
            for (int j = tid; j < n; j += group) {
                float dot = 0.0f;
                for (int i = k; i < m; i++) {
                    dot += v.get(base + i * n + k) * out.get(ob + i * n + j);
                }
                for (int i = k; i < m; i++) {
                    out.set(ob + i * n + j, out.get(ob + i * n + j) - 2.0f * v.get(base + i * n + k) * dot);
                }
            }
            kc.globalBarrier();
        }
    }

    /** The first index of round-robin pair i in round r over n2 (even) indices. */
    private static int pairA(int i, int r, int n2) {
        return i == 0 ? 0 : 1 + (i + r) % (n2 - 1);
    }

    private static int pairB(int i, int r, int n2) {
        return i == 0 ? 1 + r % (n2 - 1) : 1 + (n2 - 1 - i + r) % (n2 - 1);
    }

    /** tan of the Jacobi angle that zeroes the off-diagonal of [[app, apq], [apq, aqq]]. */
    private static float jacobiT(float app, float aqq, float apq) {
        float theta = (aqq - app) / (2.0f * apq);
        float t = 1.0f / (TornadoMath.abs(theta) + TornadoMath.sqrt(theta * theta + 1.0f));
        return theta < 0.0f ? -t : t;
    }

    /**
     * Symmetric eigendecomposition of a [B, n, n] into out [B, n + 1, n]: eigenvalues ascending in
     * row 0, the eigenvectors as the columns of rows 1..n. w, v [B, n, n], cs [B, n + 2], flag [B] and
     * perm [B, n] are scratch.
     */
    public static void eigh(KernelContext kc, FloatArray a, FloatArray out, FloatArray w, FloatArray v, FloatArray cs, IntArray flag, IntArray perm, int n, int sweeps, int group) {
        int b = kc.groupIdx;
        int tid = kc.localIdx;
        int base = b * n * n;
        int cb = b * (n + 2);
        int n2 = n + (n & 1);
        int pairs = n2 / 2;
        for (int idx = tid; idx < n * n; idx += group) {
            int i = idx / n;
            int j = idx % n;
            w.set(base + idx, a.get(base + idx));
            v.set(base + idx, i == j ? 1.0f : 0.0f);
        }
        kc.globalBarrier();
        for (int sweep = 0; sweep < sweeps; sweep++) {
            if (tid == 0) {
                flag.set(b, 0);
            }
            kc.globalBarrier();
            for (int round = 0; round < n2 - 1; round++) {
                for (int i = tid; i < pairs; i += group) {
                    int x = pairA(i, round, n2);
                    int y = pairB(i, round, n2);
                    int p = x < y ? x : y;
                    int q = x < y ? y : x;
                    float c = 1.0f;
                    float s = 0.0f;
                    if (q < n) {
                        float app = w.get(base + p * n + p);
                        float aqq = w.get(base + q * n + q);
                        float apq = w.get(base + p * n + q);
                        float scale = TornadoMath.sqrt(TornadoMath.abs(app * aqq));
                        if ((TornadoMath.abs(apq) > 1.0e-30f) & (TornadoMath.abs(apq) > 1.0e-7f * scale)) {
                            float t = jacobiT(app, aqq, apq);
                            c = 1.0f / TornadoMath.sqrt(t * t + 1.0f);
                            s = t * c;
                            flag.set(b, 1);
                        }
                    }
                    cs.set(cb + 2 * i, c);
                    cs.set(cb + 2 * i + 1, s);
                }
                kc.globalBarrier();
                for (int idx = tid; idx < pairs * n; idx += group) {
                    int i = idx / n;
                    int j = idx % n;
                    int x = pairA(i, round, n2);
                    int y = pairB(i, round, n2);
                    int p = x < y ? x : y;
                    int q = x < y ? y : x;
                    if (q < n) {
                        float c = cs.get(cb + 2 * i);
                        float s = cs.get(cb + 2 * i + 1);
                        float wp = w.get(base + p * n + j);
                        float wq = w.get(base + q * n + j);
                        w.set(base + p * n + j, c * wp - s * wq);
                        w.set(base + q * n + j, s * wp + c * wq);
                    }
                }
                kc.globalBarrier();
                for (int idx = tid; idx < pairs * n; idx += group) {
                    int i = idx / n;
                    int j = idx % n;
                    int x = pairA(i, round, n2);
                    int y = pairB(i, round, n2);
                    int p = x < y ? x : y;
                    int q = x < y ? y : x;
                    if (q < n) {
                        float c = cs.get(cb + 2 * i);
                        float s = cs.get(cb + 2 * i + 1);
                        float wp = w.get(base + j * n + p);
                        float wq = w.get(base + j * n + q);
                        w.set(base + j * n + p, c * wp - s * wq);
                        w.set(base + j * n + q, s * wp + c * wq);
                        float vp = v.get(base + j * n + p);
                        float vq = v.get(base + j * n + q);
                        v.set(base + j * n + p, c * vp - s * vq);
                        v.set(base + j * n + q, s * vp + c * vq);
                    }
                }
                kc.globalBarrier();
            }
            int rotated = flag.get(b);
            kc.globalBarrier();
            if (rotated == 0) {
                sweep = sweeps;
            }
        }
        if (tid == 0) {
            for (int i = 0; i < n; i++) {
                perm.set(b * n + i, i);
            }
            for (int i = 0; i < n; i++) {
                int best = i;
                for (int j = i + 1; j < n; j++) {
                    int pj = perm.get(b * n + j);
                    int pb = perm.get(b * n + best);
                    if (w.get(base + pj * n + pj) < w.get(base + pb * n + pb)) {
                        best = j;
                    }
                }
                int t = perm.get(b * n + i);
                perm.set(b * n + i, perm.get(b * n + best));
                perm.set(b * n + best, t);
            }
        }
        kc.globalBarrier();
        int ob = b * (n + 1) * n;
        for (int i = tid; i < n; i += group) {
            int col = perm.get(b * n + i);
            out.set(ob + i, w.get(base + col * n + col));
            float big = 0.0f;
            float sgn = 1.0f;
            for (int r = 0; r < n; r++) {
                float x = v.get(base + r * n + col);
                if (TornadoMath.abs(x) > big) {
                    big = TornadoMath.abs(x);
                    sgn = x < 0.0f ? -1.0f : 1.0f;
                }
            }
            for (int r = 0; r < n; r++) {
                out.set(ob + (1 + r) * n + i, sgn * v.get(base + r * n + col));
            }
        }
    }

    /**
     * Reduced SVD of a [B, m, n], m >= n, by one-sided Jacobi, into out [B, m + 1 + n, n]: U in rows
     * 0..m-1, the singular values (descending) in row m, V in rows m+1..m+n. The signs make the
     * largest-magnitude component of each column of U (signOnV = 0) or V (1) positive. u [B, m, n],
     * v [B, n, n], cs [B, n + 2], flag [B] and perm [B, n] are scratch.
     */
    public static void svd(KernelContext kc, FloatArray a, FloatArray out, FloatArray u, FloatArray v, FloatArray cs, IntArray flag, IntArray perm, int m, int n, int sweeps,
            int signOnV, int group) {
        int b = kc.groupIdx;
        int tid = kc.localIdx;
        int ub = b * m * n;
        int vb = b * n * n;
        int cb = b * (n + 2);
        int n2 = n + (n & 1);
        int pairs = n2 / 2;
        for (int idx = tid; idx < m * n; idx += group) {
            u.set(ub + idx, a.get(ub + idx));
        }
        for (int idx = tid; idx < n * n; idx += group) {
            v.set(vb + idx, idx / n == idx % n ? 1.0f : 0.0f);
        }
        kc.globalBarrier();
        for (int sweep = 0; sweep < sweeps; sweep++) {
            if (tid == 0) {
                flag.set(b, 0);
            }
            kc.globalBarrier();
            for (int round = 0; round < n2 - 1; round++) {
                for (int i = tid; i < pairs; i += group) {
                    int x = pairA(i, round, n2);
                    int y = pairB(i, round, n2);
                    int p = x < y ? x : y;
                    int q = x < y ? y : x;
                    float c = 1.0f;
                    float s = 0.0f;
                    if (q < n) {
                        float alpha = 0.0f;
                        float beta = 0.0f;
                        float gamma = 0.0f;
                        for (int r = 0; r < m; r++) {
                            float up = u.get(ub + r * n + p);
                            float uq = u.get(ub + r * n + q);
                            alpha += up * up;
                            beta += uq * uq;
                            gamma += up * uq;
                        }
                        float scale = TornadoMath.sqrt(alpha * beta);
                        if ((TornadoMath.abs(gamma) > 1.0e-30f) & (TornadoMath.abs(gamma) > 1.0e-7f * scale)) {
                            float t = jacobiT(alpha, beta, gamma);
                            c = 1.0f / TornadoMath.sqrt(t * t + 1.0f);
                            s = t * c;
                            flag.set(b, 1);
                        }
                    }
                    cs.set(cb + 2 * i, c);
                    cs.set(cb + 2 * i + 1, s);
                }
                kc.globalBarrier();
                int rows = m + n;
                for (int idx = tid; idx < pairs * rows; idx += group) {
                    int i = idx / rows;
                    int r = idx % rows;
                    int x = pairA(i, round, n2);
                    int y = pairB(i, round, n2);
                    int p = x < y ? x : y;
                    int q = x < y ? y : x;
                    if (q < n) {
                        float c = cs.get(cb + 2 * i);
                        float s = cs.get(cb + 2 * i + 1);
                        if (r < m) {
                            float xp = u.get(ub + r * n + p);
                            float xq = u.get(ub + r * n + q);
                            u.set(ub + r * n + p, c * xp - s * xq);
                            u.set(ub + r * n + q, s * xp + c * xq);
                        } else {
                            int rv = r - m;
                            float xp = v.get(vb + rv * n + p);
                            float xq = v.get(vb + rv * n + q);
                            v.set(vb + rv * n + p, c * xp - s * xq);
                            v.set(vb + rv * n + q, s * xp + c * xq);
                        }
                    }
                }
                kc.globalBarrier();
            }
            int rotated = flag.get(b);
            kc.globalBarrier();
            if (rotated == 0) {
                sweep = sweeps;
            }
        }
        // singular values into cs (scratch), then the descending order
        for (int j = tid; j < n; j += group) {
            float s = 0.0f;
            for (int r = 0; r < m; r++) {
                float x = u.get(ub + r * n + j);
                s += x * x;
            }
            cs.set(cb + j, TornadoMath.sqrt(s));
        }
        kc.globalBarrier();
        if (tid == 0) {
            for (int i = 0; i < n; i++) {
                perm.set(b * n + i, i);
            }
            for (int i = 0; i < n; i++) {
                int best = i;
                for (int j = i + 1; j < n; j++) {
                    if (cs.get(cb + perm.get(b * n + j)) > cs.get(cb + perm.get(b * n + best))) {
                        best = j;
                    }
                }
                int t = perm.get(b * n + i);
                perm.set(b * n + i, perm.get(b * n + best));
                perm.set(b * n + best, t);
            }
        }
        kc.globalBarrier();
        int ob = b * (m + 1 + n) * n;
        for (int i = tid; i < n; i += group) {
            int col = perm.get(b * n + i);
            float sv = cs.get(cb + col);
            float inv = sv > 0.0f ? 1.0f / sv : 0.0f;
            float big = 0.0f;
            float sgn = 1.0f;
            if (signOnV == 0) {
                for (int r = 0; r < m; r++) {
                    float x = u.get(ub + r * n + col);
                    if (TornadoMath.abs(x) > big) {
                        big = TornadoMath.abs(x);
                        sgn = x < 0.0f ? -1.0f : 1.0f;
                    }
                }
            } else {
                for (int r = 0; r < n; r++) {
                    float x = v.get(vb + r * n + col);
                    if (TornadoMath.abs(x) > big) {
                        big = TornadoMath.abs(x);
                        sgn = x < 0.0f ? -1.0f : 1.0f;
                    }
                }
            }
            for (int r = 0; r < m; r++) {
                out.set(ob + r * n + i, sgn * inv * u.get(ub + r * n + col));
            }
            out.set(ob + m * n + i, sv);
            for (int r = 0; r < n; r++) {
                out.set(ob + (m + 1 + r) * n + i, sgn * v.get(vb + r * n + col));
            }
        }
    }
}
