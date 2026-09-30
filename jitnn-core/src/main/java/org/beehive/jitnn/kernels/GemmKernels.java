package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Baseline FP32 GEMM: C[M,N] = op(A) * op(B), row-major, where op(A) is [M,K] and op(B) is [K,N].
 * {@code transA != 0} means A is stored [K,M]; {@code transB != 0} means B is stored [N,K].
 *
 * <p>16x16 tiles staged in local memory; each thread computes one element of C. This is the
 * portable baseline, not the tuned kernel: register blocking and double buffering come in the
 * wave 1 performance pass.
 */
public final class GemmKernels {

    public static final int TILE = 16;

    private GemmKernels() {
    }

    public static void gemm(KernelContext kc, FloatArray a, FloatArray b, FloatArray c, int m, int n, int k, int transA, int transB) {
        float[] as = kc.allocateFloatLocalArray(TILE * TILE);
        float[] bs = kc.allocateFloatLocalArray(TILE * TILE);
        int lc = kc.localIdx;
        int lr = kc.localIdy;
        int col = kc.groupIdx * TILE + lc;
        int row = kc.groupIdy * TILE + lr;

        float acc = 0.0f;
        int tiles = (k + TILE - 1) / TILE;
        for (int t = 0; t < tiles; t++) {
            int ka = t * TILE + lc;
            float va = 0.0f;
            if ((row < m) & (ka < k)) {
                va = transA == 0 ? a.get(row * k + ka) : a.get(ka * m + row);
            }
            as[lr * TILE + lc] = va;

            int kb = t * TILE + lr;
            float vb = 0.0f;
            if ((kb < k) & (col < n)) {
                vb = transB == 0 ? b.get(kb * n + col) : b.get(col * k + kb);
            }
            bs[lr * TILE + lc] = vb;
            kc.localBarrier();

            for (int q = 0; q < TILE; q++) {
                acc += as[lr * TILE + q] * bs[q * TILE + lc];
            }
            kc.localBarrier();
        }
        if ((row < m) & (col < n)) {
            c.set(row * n + col, acc);
        }
    }
}
