package org.beehive.jitnn.kernels;

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
}
