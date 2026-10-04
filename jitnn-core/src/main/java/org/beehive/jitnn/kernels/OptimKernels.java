package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * In-place optimiser updates. Hyper-parameters that change between steps (learning rate, bias
 * corrections) come from a small device array, so a cached plan can be replayed every step. A
 * non-zero {@code gate[0]} skips the update (a loss scaler found an overflow in the gradients).
 */
public final class OptimKernels {

    private OptimKernels() {
    }

    /** SGD without momentum. hyper = [lr, momentum, weightDecay, nesterov]. */
    public static void sgd(KernelContext kc, FloatArray p, FloatArray g, FloatArray hyper, FloatArray gate, int n) {
        int i = kc.globalIdx;
        if ((i < n) & (gate.get(0) == 0.0f)) {
            float w = p.get(i);
            float grad = g.get(i) + hyper.get(2) * w;
            p.set(i, w - hyper.get(0) * grad);
        }
    }

    /** SGD with (optionally Nesterov) momentum, as in PyTorch with dampening 0. */
    public static void sgdMomentum(KernelContext kc, FloatArray p, FloatArray g, FloatArray buf, FloatArray hyper, FloatArray gate, int n) {
        int i = kc.globalIdx;
        if ((i < n) & (gate.get(0) == 0.0f)) {
            float w = p.get(i);
            float mu = hyper.get(1);
            float grad = g.get(i) + hyper.get(2) * w;
            float v = mu * buf.get(i) + grad;
            buf.set(i, v);
            float step = hyper.get(3) != 0.0f ? grad + mu * v : v;
            p.set(i, w - hyper.get(0) * step);
        }
    }

    /**
     * Adam, or AdamW when hyper[7] != 0. hyper = [lr, beta1, beta2, eps, 1 - beta1^t,
     * 1 - beta2^t, weightDecay, decoupled].
     */
    public static void adam(KernelContext kc, FloatArray p, FloatArray g, FloatArray m, FloatArray v, FloatArray hyper, FloatArray gate, int n) {
        int i = kc.globalIdx;
        if ((i < n) & (gate.get(0) == 0.0f)) {
            float lr = hyper.get(0);
            float b1 = hyper.get(1);
            float b2 = hyper.get(2);
            float wd = hyper.get(6);
            float w = p.get(i);
            float grad = g.get(i);
            if (hyper.get(7) != 0.0f) {
                w = w - lr * wd * w;
            } else {
                grad = grad + wd * w;
            }
            float mi = b1 * m.get(i) + (1.0f - b1) * grad;
            float vi = b2 * v.get(i) + (1.0f - b2) * grad * grad;
            m.set(i, mi);
            v.set(i, vi);
            float mhat = mi / hyper.get(4);
            float vhat = vi / hyper.get(5);
            p.set(i, w - lr * mhat / (TornadoMath.sqrt(vhat) + hyper.get(3)));
        }
    }
}
