package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * BatchNorm over NCHW: statistics per channel across (n, h, w). One work-group of 256 threads per
 * channel for the reductions; element-wise kernels for the normalisation. LayerNorm and RMSNorm
 * over the last axis: one work-group per row.
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

    /** Row-norm modes: LayerNorm subtracts the mean, RMSNorm does not. */
    public static final int LAYER = 0;
    public static final int RMS = 1;
    /** Rows summed per partial in {@link #rowNormParamPartials}. */
    public static final int PARAM_CHUNK = 64;


    // The reductions are written out in each kernel: Metal cannot pass a local array to a helper
    // method (WORKAROUND J12).

    /**
     * y = (x - mean) * rstd * w + b per row (mean = 0 for RMSNorm); also stores the row mean and
     * rstd = 1 / sqrt(var + eps), with var the biased variance (or the mean square for RMSNorm).
     */
    public static void rowNormForward(KernelContext kc, FloatArray x, FloatArray w, FloatArray b, FloatArray y, FloatArray mean, FloatArray rstd, int cols, float eps, int mode, int hasBias) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        float mu = 0.0f;
        if (mode == LAYER) {
            float sum = 0.0f;
            for (int c = l; c < cols; c += GROUP) {
                sum += x.get(base + c);
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
            kc.localBarrier();
            mu = total / cols;
        }
        float sq = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            float d = x.get(base + c) - mu;
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
        float sqTotal = buf[0];
        kc.localBarrier();
        float r = 1.0f / TornadoMath.sqrt(sqTotal / cols + eps);
        for (int c = l; c < cols; c += GROUP) {
            float v = (x.get(base + c) - mu) * r * w.get(c);
            if (hasBias != 0) {
                v += b.get(c);
            }
            y.set(base + c, v);
        }
        if (l == 0) {
            mean.set(row, mu);
            rstd.set(row, r);
        }
    }

    /**
     * dx per row, with g = dy * w and xhat = (x - mean) * rstd:
     * LayerNorm dx = rstd * (g - mean(g) - xhat * mean(g * xhat)); RMSNorm drops the mean(g) term.
     */
    public static void rowNormBackward(KernelContext kc, FloatArray dy, FloatArray x, FloatArray w, FloatArray mean, FloatArray rstd, FloatArray dx, int cols, int mode) {
        float[] buf = kc.allocateFloatLocalArray(GROUP);
        int row = kc.groupIdx;
        int l = kc.localIdx;
        int base = row * cols;
        float mu = mean.get(row);
        float r = rstd.get(row);
        float sa = 0.0f;
        float sb = 0.0f;
        for (int c = l; c < cols; c += GROUP) {
            float g = dy.get(base + c) * w.get(c);
            sa += g;
            sb += g * (x.get(base + c) - mu) * r;
        }
        buf[l] = sa;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float totalA = buf[0];
        kc.localBarrier();
        buf[l] = sb;
        kc.localBarrier();
        for (int s = GROUP / 2; s > 0; s = s / 2) {
            if (l < s) {
                buf[l] = buf[l] + buf[l + s];
            }
            kc.localBarrier();
        }
        float totalB = buf[0];
        kc.localBarrier();
        float meanG = totalA / cols;
        float meanGX = totalB / cols;
        if (mode != LAYER) {
            meanG = 0.0f;
        }
        for (int c = l; c < cols; c += GROUP) {
            float xhat = (x.get(base + c) - mu) * r;
            float g = dy.get(base + c) * w.get(c);
            dx.set(base + c, r * (g - meanG - xhat * meanGX));
        }
    }

    /**
     * Partial parameter gradients over chunks of {@link #PARAM_CHUNK} rows: partW[k, c] = sum of
     * dy * xhat and partB[k, c] = sum of dy over the rows of chunk k. n = chunks * cols.
     */
    public static void rowNormParamPartials(KernelContext kc, FloatArray dy, FloatArray x, FloatArray mean, FloatArray rstd, FloatArray partW, FloatArray partB, int rows, int cols, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int k = i / cols;
            int c = i - k * cols;
            int r0 = k * PARAM_CHUNK;
            int r1 = r0 + PARAM_CHUNK < rows ? r0 + PARAM_CHUNK : rows;
            float sw = 0.0f;
            float sb = 0.0f;
            for (int r = r0; r < r1; r++) {
                float g = dy.get(r * cols + c);
                sw += g * (x.get(r * cols + c) - mean.get(r)) * rstd.get(r);
                sb += g;
            }
            partW.set(i, sw);
            partB.set(i, sb);
        }
    }

    /** out[c] = sum over k of part[k, c], in order (deterministic). */
    public static void sumPartials(KernelContext kc, FloatArray part, FloatArray out, int chunks, int cols) {
        int c = kc.globalIdx;
        if (c < cols) {
            float s = 0.0f;
            for (int k = 0; k < chunks; k++) {
                s += part.get(k * cols + c);
            }
            out.set(c, s);
        }
    }
}
