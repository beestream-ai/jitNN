package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Constant folding for inference: parameters of one layer folded into the layer before it. */
public final class FoldKernels {

    private FoldKernels() {
    }

    /**
     * BatchNorm with running statistics folded into the convolution before it: with
     * s = gamma / sqrt(var + eps), w' = w s (per output channel) and b' = beta + (b - mean) s. One
     * thread per weight; the first {@code cout} threads also write the bias. {@code per} =
     * weights per output channel; {@code hasBias} 0 means b = 0.
     */
    public static void batchNorm(KernelContext kc, FloatArray w, FloatArray bias, FloatArray gamma, FloatArray beta, FloatArray mean, FloatArray var, FloatArray wOut,
            FloatArray bOut, int cout, int per, int hasBias, float eps) {
        int i = kc.globalIdx;
        if (i < cout * per) {
            int co = i / per;
            float s = gamma.get(co) / TornadoMath.sqrt(var.get(co) + eps);
            wOut.set(i, w.get(i) * s);
            if (i < cout) {
                float si = gamma.get(i) / TornadoMath.sqrt(var.get(i) + eps);
                float b = hasBias != 0 ? bias.get(i) : 0.0f;
                bOut.set(i, beta.get(i) + (b - mean.get(i)) * si);
            }
        }
    }
}
