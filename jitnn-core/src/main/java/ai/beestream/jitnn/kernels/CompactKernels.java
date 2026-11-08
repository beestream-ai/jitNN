package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Stream compaction for operators whose output size depends on the data (nonzero, masked select,
 * unique): a flag per element, its exclusive prefix sum (the output slot), and the total. The scan
 * runs in three launches: a work-group scan of each block in local memory, a scan of the block
 * sums, and the block offsets added back. The host reads only the total, to size the output.
 */
public final class CompactKernels {

    public static final int BLOCK = 256;
    /** Flag elements that are not zero (NaN counts, as in PyTorch). */
    public static final int NONZERO = 0;
    /** Flag the first element of each run of equal values (x sorted): unique's starts. */
    public static final int RUN_START = 1;

    private CompactKernels() {
    }

    /** flags, the exclusive prefix within each block of BLOCK, and each block's sum. */
    public static void blockScan(KernelContext kc, FloatArray x, IntArray flags, IntArray offsets, IntArray blockSums, int mode, int n) {
        int[] s = kc.allocateIntLocalArray(BLOCK);
        int i = kc.globalIdx;
        int t = kc.localIdx;
        int f = 0;
        if (i < n) {
            float v = x.get(i);
            if (mode == NONZERO) {
                f = v != 0.0f ? 1 : 0;
            } else {
                f = 1;
                if (i > 0) {
                    f = v != x.get(i - 1) ? 1 : 0;
                }
            }
            flags.set(i, f);
        }
        s[t] = f;
        kc.localBarrier();
        for (int d = 1; d < BLOCK; d <<= 1) {
            int add = 0;
            if (t >= d) {
                add = s[t - d];
            }
            kc.localBarrier();
            s[t] += add;
            kc.localBarrier();
        }
        if (i < n) {
            offsets.set(i, s[t] - f);
        }
        if (t == BLOCK - 1) {
            blockSums.set(kc.groupIdx, s[t]);
        }
    }

    /** Exclusive scan of the block sums in place (one thread: the block count is n / BLOCK), and the total. */
    public static void scanBlocks(KernelContext kc, IntArray blockSums, IntArray total, int blocks) {
        if (kc.globalIdx == 0) {
            int run = 0;
            for (int b = 0; b < blocks; b++) {
                int v = blockSums.get(b);
                blockSums.set(b, run);
                run += v;
            }
            total.set(0, run);
        }
    }

    public static void addBlockOffsets(KernelContext kc, IntArray offsets, IntArray blockSums, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            offsets.set(i, offsets.get(i) + blockSums.get(i / BLOCK));
        }
    }

    /** out[offsets[i]] = x[i] for flagged i (slots past {@code size} dropped). */
    public static void compactValues(KernelContext kc, FloatArray x, IntArray flags, IntArray offsets, FloatArray out, int size, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int o = offsets.get(i);
            if ((flags.get(i) == 1) & (o < size)) {
                out.set(o, x.get(i));
            }
        }
    }

    /**
     * The coordinates of flagged elements of a tensor of shape dims [rank] into out [size, rank];
     * rows past the count are filled with {@code fill}. Launched over max(n, size) threads.
     */
    public static void compactCoords(KernelContext kc, IntArray flags, IntArray offsets, IntArray dims, IntArray total, IntArray out, int rank, int size, int fill, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int o = offsets.get(i);
            if ((flags.get(i) == 1) & (o < size)) {
                int rest = i;
                for (int d = rank - 1; d >= 0; d--) {
                    int extent = dims.get(d);
                    out.set(o * rank + d, rest % extent);
                    rest = rest / extent;
                }
            }
        }
        int count = total.get(0);
        if ((i >= count) & (i < size)) {
            for (int d = 0; d < rank; d++) {
                out.set(i * rank + d, fill);
            }
        }
    }

    /** gx[i] = flagged ? g[offsets[i]] : 0, the gradient of masked select. */
    public static void expand(KernelContext kc, FloatArray g, IntArray flags, IntArray offsets, FloatArray gx, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float v = 0.0f;
            if (flags.get(i) == 1) {
                v = g.get(offsets.get(i));
            }
            gx.set(i, v);
        }
    }

    /** counts[k] = length of the k-th run of equal sorted values. */
    public static void runLengths(KernelContext kc, IntArray flags, IntArray offsets, IntArray counts, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            if (flags.get(i) == 1) {
                int j = i + 1;
                boolean more = true;
                while (more & (j < n)) {
                    if (flags.get(j) == 1) {
                        more = false;
                    } else {
                        j++;
                    }
                }
                counts.set(offsets.get(i), j - i);
            }
        }
    }

    /** inverse[j] = the unique index of x[j]: the run holding its sorted position ranks[j]. */
    public static void inverse(KernelContext kc, IntArray ranks, IntArray flags, IntArray offsets, IntArray inverse, int n) {
        int j = kc.globalIdx;
        if (j < n) {
            int p = ranks.get(j);
            inverse.set(j, offsets.get(p) + flags.get(p) - 1);
        }
    }
}
