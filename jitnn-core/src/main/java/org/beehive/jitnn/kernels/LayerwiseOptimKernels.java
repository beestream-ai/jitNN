package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Layer-wise adaptive optimisers, LAMB (You et al., 2020) and LARS (You et al., 2017): the update of
 * one parameter is scaled by a trust ratio of norms over that whole parameter, computed between the
 * kernels here by ordinary reductions. {@code stats} = [norm, factor, skip] as for the other
 * optimisers: gradients are scaled by the factor and nothing changes when skip is non-zero.
 */
public final class LayerwiseOptimKernels {

    private LayerwiseOptimKernels() {
    }

    /**
     * LAMB's moments and raw update: m and v as Adam's, u = m^ / (sqrt(v^) + eps) + weightDecay p.
     * hyper = [lr, beta1, beta2, eps, 1 - beta1^t, 1 - beta2^t, weightDecay].
     */
    public static void lambMoments(KernelContext kc, FloatArray p, FloatArray g, FloatArray m, FloatArray v, FloatArray u, FloatArray hyper, FloatArray stats, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            if (stats.get(2) == 0.0f) {
                float grad = g.get(i) * stats.get(1);
                float b1 = hyper.get(1);
                float b2 = hyper.get(2);
                float mi = b1 * m.get(i) + (1.0f - b1) * grad;
                float vi = b2 * v.get(i) + (1.0f - b2) * grad * grad;
                m.set(i, mi);
                v.set(i, vi);
                u.set(i, (mi / hyper.get(4)) / (TornadoMath.sqrt(vi / hyper.get(5)) + hyper.get(3)) + hyper.get(6) * p.get(i));
            } else {
                u.set(i, 0.0f);
            }
        }
    }

    /** p -= lr r u with the trust ratio r = |p| / |u| (1 when either norm is 0); norms = [|p|, |u|]. */
    public static void trustUpdate(KernelContext kc, FloatArray p, FloatArray u, FloatArray norms, float lr, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float wn = norms.get(0);
            float un = norms.get(1);
            float r = (wn > 0.0f) & (un > 0.0f) ? wn / un : 1.0f;
            p.set(i, p.get(i) - lr * r * u.get(i));
        }
    }

    /**
     * LARS with momentum: local = eta |p| / (|g| + weightDecay |p| + eps) (1 when a norm is 0),
     * buf = momentum buf + lr local (g + weightDecay p), p -= buf. norms = [|p|, |g| before the
     * factor]; hyper = [lr, momentum, weightDecay, eta, eps].
     */
    public static void larsUpdate(KernelContext kc, FloatArray p, FloatArray g, FloatArray buf, FloatArray norms, FloatArray hyper, FloatArray stats, int n) {
        int i = kc.globalIdx;
        if ((i < n) & (stats.get(2) == 0.0f)) {
            float factor = stats.get(1);
            float wd = hyper.get(2);
            float wn = norms.get(0);
            float gn = norms.get(1) * factor;
            float local = (wn > 0.0f) & (gn > 0.0f) ? hyper.get(3) * wn / (gn + wd * wn + hyper.get(4)) : 1.0f;
            float w = p.get(i);
            float b = hyper.get(1) * buf.get(i) + hyper.get(0) * local * (g.get(i) * factor + wd * w);
            buf.set(i, b);
            p.set(i, w - b);
        }
    }
}
