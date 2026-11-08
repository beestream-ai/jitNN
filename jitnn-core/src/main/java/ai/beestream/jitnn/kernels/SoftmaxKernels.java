package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Row-wise softmax, log-softmax, their gradients, and the negative log-likelihood loss. One
 * work-group of 256 threads per row; row statistics are reduced in local memory.
 */
public final class SoftmaxKernels {

    private static final int GROUP = 256;

    private SoftmaxKernels() {
    }

    /** y = softmax(x) per row, or log-softmax when {@code logMode != 0}. */
    public static void softmaxRows(KernelContext kc, FloatArray x, FloatArray y, int cols, int logMode) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;

        float m = Float.NEGATIVE_INFINITY;
        for (int c = l; c < cols; c += GROUP) {
            float v = x.get(base + c);
            m = v > m ? v : m;
        }
        buf[l] = m;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                float o = buf[l + s];
                buf[l] = o > buf[l] ? o : buf[l];
            }
            kc.localBarrier();
        }
        float rowMax = buf[0];
        kc.localBarrier();

        float sum = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            sum += TornadoMath.exp(x.get(base + c) - rowMax);
        }
        buf[l] = sum;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float total = buf[0];

        if (logMode != 0) {
            float logTotal = TornadoMath.log(total);
            for (int c = l; c < cols; c += GROUP) {
                y.set(base + c, x.get(base + c) - rowMax - logTotal);
            }
        } else {
            float inv = 1.0f / total;
            for (int c = l; c < cols; c += GROUP) {
                y.set(base + c, TornadoMath.exp(x.get(base + c) - rowMax) * inv);
            }
        }
    }

    /** Longest row {@link #softmaxRowsCached} keeps in local memory (16 KB). */
    public static final int CACHED_COLS = 4096;

    /**
     * Softmax / log-softmax for rows of at most {@link #CACHED_COLS}: the row is read from global
     * memory once into local memory, and exp(x - max) is kept there between the sum and the write.
     */
    public static void softmaxRowsCached(KernelContext kc, FloatArray x, FloatArray y, int cols, int logMode) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        float[] row = kc.allocateFloatLocalArray(CACHED_COLS);
        int r = kc.groupIdx;
        int l = kc.localIdx;
        int base = r * cols;
        float m = Float.NEGATIVE_INFINITY;
        for (int c = l; c < cols; c += GROUP) {
            float v = x.get(base + c);
            row[c] = v;
            m = v > m ? v : m;
        }
        buf[l] = m;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                float o = buf[l + s];
                buf[l] = o > buf[l] ? o : buf[l];
            }
            kc.localBarrier();
        }
        float rowMax = buf[0];
        kc.localBarrier();
        float sum = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            float ex = TornadoMath.exp(row[c] - rowMax);
            if (logMode == 0) {
                row[c] = ex;
            }
            sum += ex;
        }
        buf[l] = sum;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float total = buf[0];
        if (logMode != 0) {
            float shift = rowMax + TornadoMath.log(total);
            for (int c = l; c < cols; c += GROUP) {
                y.set(base + c, row[c] - shift);
            }
        } else {
            float inv = 1.0f / total;
            for (int c = l; c < cols; c += GROUP) {
                y.set(base + c, row[c] * inv);
            }
        }
    }

    /**
     * Gradient of softmax (dx = y * (dy - sum(dy * y))) or of log-softmax
     * (dx = dy - exp(y) * sum(dy)), per row. {@code y} is the forward output.
     */
    public static void softmaxBackwardRows(KernelContext kc, FloatArray y, FloatArray dy, FloatArray dx, int cols, int logMode) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;

        float acc = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            float g = dy.get(base + c);
            acc += logMode != 0 ? g : g * y.get(base + c);
        }
        buf[l] = acc;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float dot = buf[0];

        for (int c = l; c < cols; c += GROUP) {
            float g = dy.get(base + c);
            float v = y.get(base + c);
            float r;
            if (logMode != 0) {
                r = g - TornadoMath.exp(v) * dot;
            } else {
                r = v * (g - dot);
            }
            dx.set(base + c, r);
        }
    }

    /**
     * Attention probabilities per row of scores: y = softmax(scale * x) over the first {@code valid}
     * columns and 0 beyond, where valid = cols, or with {@code causal} the keys query t may see,
     * 0 .. t (PyTorch's is_causal, aligned to the top left), t = row % queryLen. One work-group
     * per row.
     */
    public static void attentionSoftmaxRows(KernelContext kc, FloatArray x, FloatArray y, int cols, float scale, int causal, int queryLen) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        int valid = cols;
        if (causal != 0) {
            int limit = row % queryLen + 1;
            valid = limit < cols ? limit : cols;
        }
        float m = Float.NEGATIVE_INFINITY;
        for (int c = l; c < valid; c += GROUP) {
            float v = scale * x.get(base + c);
            m = v > m ? v : m;
        }
        buf[l] = m;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                float o = buf[l + s];
                buf[l] = o > buf[l] ? o : buf[l];
            }
            kc.localBarrier();
        }
        float rowMax = buf[0];
        kc.localBarrier();
        float sum = 0.0f;
        for (int c = l; c < valid; c += GROUP) {
            sum += TornadoMath.exp(scale * x.get(base + c) - rowMax);
        }
        buf[l] = sum;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float inv = 1.0f / buf[0];
        for (int c = l; c < cols; c += GROUP) {
            float p = 0.0f;
            if (c < valid) {
                p = TornadoMath.exp(scale * x.get(base + c) - rowMax) * inv;
            }
            y.set(base + c, p);
        }
    }

    /** Gradient of {@link #attentionSoftmaxRows}: dx = scale * y * (dy - sum(dy * y)); masked entries have y = 0. */
    public static void attentionSoftmaxBackwardRows(KernelContext kc, FloatArray y, FloatArray dy, FloatArray dx, int cols, float scale) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        float acc = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            acc += dy.get(base + c) * y.get(base + c);
        }
        buf[l] = acc;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float dot = buf[0];
        for (int c = l; c < cols; c += GROUP) {
            float v = y.get(base + c);
            dx.set(base + c, scale * v * (dy.get(base + c) - dot));
        }
    }

    /** loss[r] = -logp[r, label[r]], one thread per row. */
    public static void nllRows(KernelContext kc, FloatArray logp, IntArray labels, FloatArray loss, int rows, int cols) {
        int r = kc.globalIdx;
        if (r < rows) {
            loss.set(r, -logp.get(r * cols + labels.get(r)));
        }
    }

    /** dlogp[r, c] = -dloss[r] where c == label[r], else 0; one thread per element. */
    public static void nllBackward(KernelContext kc, IntArray labels, FloatArray dloss, FloatArray dlogp, int rows, int cols) {
        int i = kc.globalIdx;
        if (i < rows * cols) {
            int r = i / cols;
            int c = i - r * cols;
            dlogp.set(i, c == labels.get(r) ? -dloss.get(r) : 0.0f);
        }
    }

    // WORKAROUND(TornadoVM): the row statistics are written out in each kernel. Passing the
    // local-memory array to a shared helper method fails to sketch on Metal ("-1|LocalArray").

    /** Fused cross-entropy, per row: loss[r] = logsumexp(x[r]) - x[r, label[r]]. One work-group per row. */
    public static void crossEntropyRows(KernelContext kc, FloatArray x, IntArray labels, FloatArray loss, int cols) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        float max = Float.NEGATIVE_INFINITY;
        for (int c = l; c < cols; c += GROUP) {
            float v = x.get(base + c);
            max = v > max ? v : max;
        }
        buf[l] = max;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                float o = buf[l + s];
                buf[l] = o > buf[l] ? o : buf[l];
            }
            kc.localBarrier();
        }
        max = buf[0];
        kc.localBarrier();
        float sum = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            sum += TornadoMath.exp(x.get(base + c) - max);
        }
        buf[l] = sum;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        sum = buf[0];
        if (l == 0) {
            loss.set(row, TornadoMath.log(sum) + max - x.get(base + labels.get(row)));
        }
    }

    /**
     * Gradient of the mean cross-entropy: dx[r, c] = (softmax(x)[r, c] - [c == label[r]]) * g[0] * scale,
     * with scale = 1 / rows. One work-group per row.
     */
    public static void crossEntropyBackwardRows(KernelContext kc, FloatArray x, IntArray labels, FloatArray g, FloatArray dx, int cols, float scale) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        float max = Float.NEGATIVE_INFINITY;
        for (int c = l; c < cols; c += GROUP) {
            float v = x.get(base + c);
            max = v > max ? v : max;
        }
        buf[l] = max;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                float o = buf[l + s];
                buf[l] = o > buf[l] ? o : buf[l];
            }
            kc.localBarrier();
        }
        max = buf[0];
        kc.localBarrier();
        float sum = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            sum += TornadoMath.exp(x.get(base + c) - max);
        }
        buf[l] = sum;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        sum = buf[0];
        float inv = 1.0f / sum;
        float gs = g.get(0) * scale;
        int label = labels.get(row);
        for (int c = l; c < cols; c += GROUP) {
            float p = TornadoMath.exp(x.get(base + c) - max) * inv;
            dx.set(base + c, (p - (c == label ? 1.0f : 0.0f)) * gs);
        }
    }
}
