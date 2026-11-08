package ai.beestream.jitnn.kernels;

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
    public static final int PROD = 3;

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
        } else if (op == PROD) {
            r = 1.0f;
        } else {
            r = 0.0f;
        }
        return r;
    }

    /**
     * Folds v into acc. Max and min propagate NaN, as PyTorch's: a NaN anywhere makes the result NaN
     * (bit-pattern tests, as TornadoVM folds {@code x != x} away).
     */
    public static float combine(int op, float acc, float v) {
        float r;
        if (op == MAX) {
            r = ElementwiseKernels.isNaN(acc) ? acc : (ElementwiseKernels.isNaN(v) || v > acc ? v : acc);
        } else if (op == MIN) {
            r = ElementwiseKernels.isNaN(acc) ? acc : (ElementwiseKernels.isNaN(v) || v < acc ? v : acc);
        } else if (op == PROD) {
            r = acc * v;
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

    /** Index of the first maximum along the axis, one thread per output. */
    /** Index of the first minimum along the axis, one thread per output. */
    public static void argminThread(KernelContext kc, FloatArray x, IntArray out, int outer, int len, int inner) {
        int i = kc.globalIdx;
        if (i < outer * inner) {
            int o = i / inner;
            int j = i - o * inner;
            int base = o * len * inner + j;
            float best = x.get(base);
            int bestK = 0;
            for (int k = 1; k < len; k++) {
                float v = x.get(base + k * inner);
                // the first NaN wins, as in PyTorch
                if (!ElementwiseKernels.isNaN(best) && (ElementwiseKernels.isNaN(v) || v < best)) {
                    best = v;
                    bestK = k;
                }
            }
            out.set(i, bestK);
        }
    }

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
                if (!ElementwiseKernels.isNaN(best) && (ElementwiseKernels.isNaN(v) || v > best)) {
                    best = v;
                    bestK = k;
                }
            }
            out.set(i, bestK);
        }
    }
}
