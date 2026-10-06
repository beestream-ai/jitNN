package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Scans and sorting along one axis of a tensor seen as [outer, n, inner]: line (o, q) is the n
 * elements at o * n * inner + k * inner + q. Every kernel is deterministic.
 */
public final class ScanKernels {

    public static final int SUM = 0;
    public static final int PROD = 1;

    private ScanKernels() {
    }

    /**
     * Inclusive scan of every line, one thread per line, in order; mode {@link #SUM} or
     * {@link #PROD}; {@code reverse} scans from the end (the gradient of cumsum).
     */
    public static void scan(KernelContext kc, FloatArray x, FloatArray y, int n, int inner, int lines, int mode, int reverse) {
        int l = kc.globalIdx;
        if (l < lines) {
            int o = l / inner;
            int q = l - o * inner;
            int base = o * n * inner + q;
            float acc = mode == SUM ? 0.0f : 1.0f;
            for (int t = 0; t < n; t++) {
                int k = reverse != 0 ? n - 1 - t : t;
                float v = x.get(base + k * inner);
                acc = mode == SUM ? acc + v : acc * v;
                y.set(base + k * inner, acc);
            }
        }
    }

    /**
     * Gradient of cumprod, exact with zeros in x: dx_i = sum over j >= i of dy_j times the product of
     * x_0 .. x_j without x_i. One thread per element.
     */
    public static void cumprodBackward(KernelContext kc, FloatArray x, FloatArray dy, FloatArray dx, int n, int inner, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int q = i % inner;
            int k = (i / inner) % n;
            int o = i / (inner * n);
            int base = o * n * inner + q;
            float left = 1.0f;
            for (int t = 0; t < k; t++) {
                left *= x.get(base + t * inner);
            }
            float right = 1.0f;
            float acc = 0.0f;
            for (int j = k; j < n; j++) {
                if (j > k) {
                    right *= x.get(base + j * inner);
                }
                acc += dy.get(base + j * inner) * left * right;
            }
            dx.set(i, acc);
        }
    }

    /**
     * Stable sort of every line by rank: element k goes to position rank_k = number of elements
     * that come before it (smaller, or equal with a smaller index; reversed for {@code descending}).
     * Writes the sorted values, their original indices and every element's rank. One thread per
     * element, O(n) each.
     */
    public static void sortByRank(KernelContext kc, FloatArray x, FloatArray values, IntArray indices, IntArray ranks, int n, int inner, int descending, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int q = i % inner;
            int k = (i / inner) % n;
            int o = i / (inner * n);
            int base = o * n * inner + q;
            float v = x.get(i);
            int rank = 0;
            for (int j = 0; j < n; j++) {
                float u = x.get(base + j * inner);
                boolean before = descending != 0 ? (u > v) | ((u == v) & (j < k)) : (u < v) | ((u == v) & (j < k));
                if (before) {
                    rank++;
                }
            }
            values.set(base + rank * inner, v);
            indices.set(base + rank * inner, k);
            ranks.set(i, rank);
        }
    }

    /** Gradient of the sorted values: dx at each element is dy at its rank. */
    public static void sortBackward(KernelContext kc, FloatArray dy, IntArray ranks, FloatArray dx, int n, int inner, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int q = i % inner;
            int o = i / (inner * n);
            dx.set(i, dy.get(o * n * inner + ranks.get(i) * inner + q));
        }
    }
}
