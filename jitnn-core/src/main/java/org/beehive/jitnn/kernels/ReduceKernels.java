package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Reductions along one axis. The input is viewed as {@code [outer, len, inner]} and the output as
 * {@code [outer, inner]}. Two launch shapes:
 *
 * <ul>
 * <li>{@link #reduceThread}: one thread per output, looping over the axis. Best when there are many
 * outputs (e.g. the bias gradient, summing a batch).</li>
 * <li>{@link #reduceGroup}: one work-group of 256 threads per output, with a tree reduction in
 * local memory. Best for long axes and few outputs (e.g. a loss).</li>
 * </ul>
 */
public final class ReduceKernels {

    public static final int SUM = 0;
    public static final int MAX = 1;
    public static final int MIN = 2;

    /** Must equal {@code Emitter.LOCAL}. */
    private static final int GROUP = 256;

    private ReduceKernels() {
    }

    public static float identity(int op) {
        float r;
        if (op == MAX) {
            r = Float.NEGATIVE_INFINITY;
        } else if (op == MIN) {
            r = Float.POSITIVE_INFINITY;
        } else {
            r = 0.0f;
        }
        return r;
    }

    public static float combine(int op, float acc, float v) {
        float r;
        if (op == MAX) {
            r = v > acc ? v : acc;
        } else if (op == MIN) {
            r = v < acc ? v : acc;
        } else {
            r = acc + v;
        }
        return r;
    }

    public static void reduceThread(KernelContext kc, FloatArray x, FloatArray out, int outer, int len, int inner, int op, float scale) {
        int i = kc.globalIdx;
        if (i < outer * inner) {
            int o = i / inner;
            int j = i - o * inner;
            int base = o * len * inner + j;
            float acc = identity(op);
            for (int k = 0; k < len; k++) {
                acc = combine(op, acc, x.get(base + k * inner));
            }
            out.set(i, op == SUM ? acc * scale : acc);
        }
    }

    public static void reduceGroup(KernelContext kc, FloatArray x, FloatArray out, int outer, int len, int inner, int op, float scale) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int g = kc.groupIdx;
        int l = kc.localIdx;
        int o = g / inner;
        int j = g - o * inner;
        int base = o * len * inner + j;
        float acc = identity(op);
        for (int k = l; k < len; k += GROUP) {
            acc = combine(op, acc, x.get(base + k * inner));
        }
        buf[l] = acc;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = combine(op, buf[l], buf[l + s]);
            }
            kc.localBarrier();
        }
        if (l == 0) {
            out.set(g, op == SUM ? buf[0] * scale : buf[0]);
        }
    }

    /**
     * First pass of a split reduction over a contiguous axis ({@code inner == 1}): work-group g
     * reduces chunk {@code g % parts} of row {@code g / parts} into {@code partial[g]}.
     */
    public static void reduceRowsSplit(KernelContext kc, FloatArray x, FloatArray partial, int len, int parts, int chunk, int op) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int g = kc.groupIdx;
        int l = kc.localIdx;
        int row = g / parts;
        int p = g - row * parts;
        int start = p * chunk;
        int end = start + chunk < len ? start + chunk : len;
        int base = row * len;
        float acc = identity(op);
        for (int k = start + l; k < end; k += GROUP) {
            acc = combine(op, acc, x.get(base + k));
        }
        buf[l] = acc;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = combine(op, buf[l], buf[l + s]);
            }
            kc.localBarrier();
        }
        if (l == 0) {
            partial.set(g, buf[0]);
        }
    }

    /**
     * First pass of a split reduction over a strided axis ({@code inner > 1}): thread
     * (p, o, j) reduces rows {@code [p * chunk, (p + 1) * chunk)} of column (o, j) into
     * {@code partial[p * outer * inner + o * inner + j]}. Consecutive threads read consecutive
     * columns, so loads are coalesced.
     */
    public static void reduceColumnsSplit(KernelContext kc, FloatArray x, FloatArray partial, int outer, int len, int inner, int parts, int chunk, int op) {
        int id = kc.globalIdx;
        int outputs = outer * inner;
        if (id < outputs * parts) {
            int p = id / outputs;
            int rem = id - p * outputs;
            int o = rem / inner;
            int j = rem - o * inner;
            int start = p * chunk;
            int end = start + chunk < len ? start + chunk : len;
            float acc = identity(op);
            for (int k = start; k < end; k++) {
                acc = combine(op, acc, x.get((o * len + k) * inner + j));
            }
            partial.set(id, acc);
        }
    }

    /** Columns per work-group of {@link #reduceColumnsGroup}. */
    public static final int COL_TILE = 32;
    /** Row lanes per work-group of {@link #reduceColumnsGroup}. */
    public static final int ROW_LANES = 8;

    /**
     * First pass of a split reduction over a strided axis, by work-groups: a group of 32 x 8
     * threads takes COL_TILE consecutive output columns (128-byte rows) and rows
     * {@code [p * chunk, (p + 1) * chunk)}, each of the 8 row lanes reducing every 8th row; the lanes
     * are combined in order (deterministic) into {@code partial[p * outputs + column]}. Launch
     * parts * ceil(outputs / COL_TILE) groups of 256 threads.
     */
    public static void reduceColumnsGroup(KernelContext kc, FloatArray x, FloatArray partial, int outer, int len, int inner, int parts, int chunk, int op) {
        float[] buf = kc.allocateFloatLocalArray(COL_TILE * ROW_LANES);
        int tid = kc.localIdx;
        int lc = tid % COL_TILE;
        int lr = tid / COL_TILE;
        int outputs = outer * inner;
        int tiles = (outputs + COL_TILE - 1) / COL_TILE;
        int p = kc.groupIdx / tiles;
        int col = (kc.groupIdx - p * tiles) * COL_TILE + lc;
        int start = p * chunk;
        int end = start + chunk < len ? start + chunk : len;
        float acc = identity(op);
        if (col < outputs) {
            int o = col / inner;
            int j = col - o * inner;
            for (int k = start + lr; k < end; k += ROW_LANES) {
                acc = combine(op, acc, x.get((o * len + k) * inner + j));
            }
        }
        buf[tid] = acc;
        kc.localBarrier();
        if ((lr == 0) & (col < outputs)) {
            float r = buf[lc];
            for (int q = 1; q < ROW_LANES; q++) {
                r = combine(op, r, buf[q * COL_TILE + lc]);
            }
            partial.set(p * outputs + col, r);
        }
    }

    /** Index of the first maximum along the axis, one thread per output. */
    public static void argmaxThread(KernelContext kc, FloatArray x, IntArray out, int outer, int len, int inner) {
        int i = kc.globalIdx;
        if (i < outer * inner) {
            int o = i / inner;
            int j = i - o * inner;
            int base = o * len * inner + j;
            float best = x.get(base);
            int bestK = 0;
            for (int k = 1; k < len; k++) {
                float v = x.get(base + k * inner);
                if (v > best) {
                    best = v;
                    bestK = k;
                }
            }
            out.set(i, bestK);
        }
    }
}
