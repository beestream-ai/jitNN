package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * BatchNorm over NCHW: statistics per channel across (n, h, w). One work-group of 256 threads per
 * channel for the reductions; element-wise kernels for the normalisation.
 */
public final class NormKernels {

    private static final int GROUP = 256;

    private NormKernels() {
    }

    /**
     * Per-channel mean and 1/sqrt(var + eps) (two passes, biased variance), and the running
     * statistics update: running = (1 - momentum) * running + momentum * batch, with the unbiased
     * variance, as in PyTorch.
     */
    public static void batchStats(KernelContext kc, FloatArray x, FloatArray mean, FloatArray invstd, FloatArray runMean, FloatArray runVar, int n, int c, int hw, float eps, float momentum) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int ch = kc.groupIdx;
        int l = kc.localIdx;
        int count = n * hw;
        float acc = 0.0f;
        for (int i = l; i < count; i += GROUP) {
            int b = i / hw;
            int j = i - b * hw;
            acc += x.get((b * c + ch) * hw + j);
        }
        buf[l] = acc;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float mu = buf[0] / count;
        kc.localBarrier();
        float sq = 0.0f;
        for (int i = l; i < count; i += GROUP) {
            int b = i / hw;
            int j = i - b * hw;
            float d = x.get((b * c + ch) * hw + j) - mu;
            sq += d * d;
        }
        buf[l] = sq;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        if (l == 0) {
            float var = buf[0] / count;
            mean.set(ch, mu);
            invstd.set(ch, 1.0f / TornadoMath.sqrt(var + eps));
            float unbiased = count > 1 ? buf[0] / (count - 1) : var;
            runMean.set(ch, (1.0f - momentum) * runMean.get(ch) + momentum * mu);
            runVar.set(ch, (1.0f - momentum) * runVar.get(ch) + momentum * unbiased);
        }
    }

    /** y = (x - mean[c]) * invstd[c] * gamma[c] + beta[c], with optional ReLU (bit 2 of epilogue). */
    public static void batchApply(KernelContext kc, FloatArray x, FloatArray y, FloatArray mean, FloatArray invstd, FloatArray gamma, FloatArray beta, int c, int hw, int total, int relu) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = (i / hw) % c;
            float v = (x.get(i) - mean.get(ch)) * invstd.get(ch) * gamma.get(ch) + beta.get(ch);
            if (relu != 0) {
                v = v > 0.0f ? v : 0.0f;
            }
            y.set(i, v);
        }
    }

    /** Inference: normalises with the running statistics. */
    public static void batchApplyRunning(KernelContext kc, FloatArray x, FloatArray y, FloatArray runMean, FloatArray runVar, FloatArray gamma, FloatArray beta, int c, int hw, int total, float eps, int relu) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = (i / hw) % c;
            float v = (x.get(i) - runMean.get(ch)) / TornadoMath.sqrt(runVar.get(ch) + eps) * gamma.get(ch) + beta.get(ch);
            if (relu != 0) {
                v = v > 0.0f ? v : 0.0f;
            }
            y.set(i, v);
        }
    }

    /** Per channel: dbeta = sum(dy), dgamma = sum(dy * xhat). */
    public static void batchBackwardReduce(KernelContext kc, FloatArray dy, FloatArray x, FloatArray mean, FloatArray invstd, FloatArray dgamma, FloatArray dbeta, int n, int c, int hw) {
        float[] bufA = kc.allocateFloatLocalArray(GROUP);
        float[] bufB = kc.allocateFloatLocalArray(GROUP);
        int ch = kc.groupIdx;
        int l = kc.localIdx;
        int count = n * hw;
        float mu = mean.get(ch);
        float is = invstd.get(ch);
        float sa = 0.0f;
        float sb = 0.0f;
        for (int i = l; i < count; i += GROUP) {
            int b = i / hw;
            int j = i - b * hw;
            int at = (b * c + ch) * hw + j;
            float g = dy.get(at);
            sa += g;
            sb += g * (x.get(at) - mu) * is;
        }
        bufA[l] = sa;
        bufB[l] = sb;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                bufA[l] = bufA[l] + bufA[l + s];
                bufB[l] = bufB[l] + bufB[l + s];
            }
            kc.localBarrier();
        }
        if (l == 0) {
            dbeta.set(ch, bufA[0]);
            dgamma.set(ch, bufB[0]);
        }
    }

    /** dx = gamma * invstd * (dy - dbeta / M - xhat * dgamma / M), M = n * hw. */
    public static void batchBackwardApply(KernelContext kc, FloatArray dy, FloatArray x, FloatArray mean, FloatArray invstd, FloatArray gamma, FloatArray dgamma, FloatArray dbeta, FloatArray dx, int c, int hw, int total, float invCount) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = (i / hw) % c;
            float is = invstd.get(ch);
            float xhat = (x.get(i) - mean.get(ch)) * is;
            dx.set(i, gamma.get(ch) * is * (dy.get(i) - dbeta.get(ch) * invCount - xhat * dgamma.get(ch) * invCount));
        }
    }
}
