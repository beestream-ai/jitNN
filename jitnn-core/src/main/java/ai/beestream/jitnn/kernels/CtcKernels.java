package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Connectionist temporal classification (Graves et al., 2006) in log space, PyTorch's ctc_loss
 * conventions: log-probabilities [T, B, C], targets [B, L] padded, the extended label sequence
 * blank, l1, blank, l2, ..., blank of S = 2 L_b + 1 states. One work-group per sample, one thread per
 * state (the group must hold 2 L + 1 threads), a work-group barrier with a global fence per time step; alpha and beta are kept in
 * global memory [B, T, S] for the gradient.
 */
public final class CtcKernels {

    private static final float NEG = -1.0e30f;

    private CtcKernels() {
    }

    /** log(exp(a) + exp(b)) without overflow; NEG stands for log 0. */
    public static float logAdd(float a, float b) {
        float hi = a > b ? a : b;
        float lo = a > b ? b : a;
        float r = hi;
        if (lo > NEG * 0.5f) {
            r = hi + ElementwiseKernels.log1p(TornadoMath.exp(lo - hi));
        }
        return r;
    }

    private static int label(IntArray targets, int b, int s, int lmax, int blank) {
        return (s & 1) == 0 ? blank : targets.get(b * lmax + s / 2);
    }

    /** Alphas of every sample; loss[b] = -log p(target_b | input_b). */
    public static void forward(KernelContext kc, FloatArray lp, IntArray targets, IntArray inLen, IntArray tgtLen, FloatArray alpha, FloatArray loss,
            int tmax, int batch, int classes, int lmax, int blank) {
        int b = kc.groupIdx;
        int s = kc.localIdx;
        int smax = 2 * lmax + 1;
        int tb = inLen.get(b);
        int sb = 2 * tgtLen.get(b) + 1;
        int base = b * tmax * smax;
        if (s < smax) {
            float a = NEG;
            if (s < sb) {
                if ((s == 0) | (s == 1)) {
                    a = lp.get(b * classes + label(targets, b, s, lmax, blank));
                }
            }
            alpha.set(base + s, a);
        }
        kc.globalBarrier();
        for (int t = 1; t < tmax; t++) {
            if (s < smax) {
                float a = NEG;
                if ((s < sb) & (t < tb)) {
                    int l = label(targets, b, s, lmax, blank);
                    a = alpha.get(base + (t - 1) * smax + s);
                    if (s >= 1) {
                        a = logAdd(a, alpha.get(base + (t - 1) * smax + s - 1));
                    }
                    if ((s >= 2) & (l != blank)) {
                        if (l != label(targets, b, s - 2, lmax, blank)) {
                            a = logAdd(a, alpha.get(base + (t - 1) * smax + s - 2));
                        }
                    }
                    a = a + lp.get((t * batch + b) * classes + l);
                }
                alpha.set(base + t * smax + s, a);
            }
            kc.globalBarrier();
        }
        if (s == 0) {
            int last = base + (tb - 1) * smax;
            float total = alpha.get(last + sb - 1);
            if (sb >= 2) {
                total = logAdd(total, alpha.get(last + sb - 2));
            }
            loss.set(b, -total);
        }
    }

    /**
     * Betas and the gradient with respect to the log-probabilities, scaled by g[b] (the incoming
     * gradient of each sample's loss): exp(lp) g - g / p sum over states s of class c of
     * alpha_t(s) beta_t(s). Positions past a sample's input length get 0.
     */
    public static void backward(KernelContext kc, FloatArray lp, IntArray targets, IntArray inLen, IntArray tgtLen, FloatArray alpha, FloatArray beta, FloatArray loss,
            FloatArray g, FloatArray grad, int tmax, int batch, int classes, int lmax, int blank, int group) {
        int b = kc.groupIdx;
        int s = kc.localIdx;
        int smax = 2 * lmax + 1;
        int tb = inLen.get(b);
        int sb = 2 * tgtLen.get(b) + 1;
        int base = b * tmax * smax;
        for (int t = tmax - 1; t >= 0; t--) {
            if (s < smax) {
                float v = NEG;
                if ((s < sb) & (t < tb)) {
                    int l = label(targets, b, s, lmax, blank);
                    if (t == tb - 1) {
                        if (s >= sb - 2) {
                            v = lp.get((t * batch + b) * classes + l);
                        }
                    } else {
                        v = beta.get(base + (t + 1) * smax + s);
                        if (s + 1 < sb) {
                            v = logAdd(v, beta.get(base + (t + 1) * smax + s + 1));
                        }
                        if ((s + 2 < sb) & (l != blank)) {
                            if (l != label(targets, b, s + 2, lmax, blank)) {
                                v = logAdd(v, beta.get(base + (t + 1) * smax + s + 2));
                            }
                        }
                        v = v + lp.get((t * batch + b) * classes + l);
                    }
                }
                beta.set(base + t * smax + s, v);
            }
            kc.globalBarrier();
        }
        // gradient: thread s covers classes s, s + group, ...
        float logp = -loss.get(b);
        float gb = g.get(b);
        for (int t = 0; t < tmax; t++) {
            for (int c = s; c < classes; c += group) {
                float v = 0.0f;
                if (t < tb) {
                    float acc = NEG;
                    for (int q = 0; q < sb; q++) {
                        if (label(targets, b, q, lmax, blank) == c) {
                            int at = base + t * smax + q;
                            // alpha and beta both include lp at t: divide one out
                            acc = logAdd(acc, alpha.get(at) + beta.get(at) - lp.get((t * batch + b) * classes + c));
                        }
                    }
                    float l = lp.get((t * batch + b) * classes + c);
                    v = gb * (TornadoMath.exp(l) - TornadoMath.exp(acc - logp));
                }
                grad.set((t * batch + b) * classes + c, v);
            }
        }
    }
}
