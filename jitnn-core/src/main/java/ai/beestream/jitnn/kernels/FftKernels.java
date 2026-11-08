package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Discrete Fourier transforms along the last axis, complex numbers as (re, im) pairs. One general
 * kernel serves the forward and inverse, real-to-complex, complex-to-real and adjoint transforms:
 * y_k = scale sum_{j < nIn} w_j x_j e^{+-2 pi i (k j mod n) / n} for k < nOut, with y_k = 0 for k >= n.
 * The phases come from a host-computed table twiddle [n, 2] (cos, sin), which keeps them exact and
 * avoids device sin/cos. It is a direct O(n^2) sum (one thread per output), exact in its phase.
 */
public final class FftKernels {

    /** Weights w: 1, or 2 except at j = 0 and 2 j = n (the Hermitian fold of irfft), on the inputs or on the outputs. */
    public static final int WEIGHT_NONE = 0;
    public static final int WEIGHT_IN = 1;
    public static final int WEIGHT_OUT = 2;

    private FftKernels() {
    }

    private static float fold(int j, int n) {
        return ((j == 0) | (2 * j == n)) ? 1.0f : 2.0f;
    }

    /**
     * x [rows, inStride] (realIn = 1) or [rows, inStride, 2]; y [rows, nOut] (realOut = 1, the real
     * part) or [rows, nOut, 2].
     */
    public static void dft(KernelContext kc, FloatArray x, FloatArray twiddle, FloatArray y, int inStride, int nIn, int nOut, int n, float scale, int realIn, int realOut,
            int weight, int total) {
        int id = kc.globalIdx;
        if (id < total) {
            int r = id / nOut;
            int k = id % nOut;
            float re = 0.0f;
            float im = 0.0f;
            if (k < n) {
                for (int j = 0; j < nIn; j++) {
                    int ph = (k * j) % n;
                    float c = twiddle.get(2 * ph);
                    float s = twiddle.get(2 * ph + 1);
                    float xr;
                    float xi = 0.0f;
                    if (realIn == 1) {
                        xr = x.get(r * inStride + j);
                    } else {
                        xr = x.get((r * inStride + j) * 2);
                        xi = x.get((r * inStride + j) * 2 + 1);
                    }
                    float w = weight == WEIGHT_IN ? fold(j, n) : 1.0f;
                    re += w * (xr * c - xi * s);
                    im += w * (xr * s + xi * c);
                }
                float w = weight == WEIGHT_OUT ? fold(k, n) : 1.0f;
                re = re * scale * w;
                im = im * scale * w;
            }
            if (realOut == 1) {
                y.set(id, re);
            } else {
                y.set(id * 2, re);
                y.set(id * 2 + 1, im);
            }
        }
    }
}
