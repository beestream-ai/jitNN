package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The INT8 product of {@link MmaInt8Kernels#gemm} without tensor cores (OpenCL, Metal, CUDA with
 * {@code -Djitnn.tensorCores=false}), on the same packed operands: one thread per output, INT32
 * accumulation.
 */
public final class Int8Kernels {

    private static final int TILE = 128;
    private static final int STEP = 32;

    private Int8Kernels() {
    }

    public static void gemm(KernelContext ctx, IntArray a, IntArray b, FloatArray c, int m, int n, int k, FloatArray sx, FloatArray zp, int offset, FloatArray sw,
            IntArray colsum, FloatArray bias, int hasBias, int hw) {
        int col = ctx.globalIdx;
        int row = ctx.globalIdy;
        if ((col < n) & (row < m)) {
            int kp = (k + STEP - 1) / STEP * STEP;
            int steps = kp / STEP;
            int lda = kp >> 2;
            int nb = col / TILE;
            int jj = col - nb * TILE;
            // the weight bytes of this column in the tile order: panel, column pair, which of the pair
            int colBase = (jj >> 3) * 64 + ((jj & 7) >> 1);
            int colShift = (jj & 1) << 4;
            int acc = 0;
            for (int l = 0; l < k; l++) {
                int av = (a.get(row * lda + (l >> 2)) << (24 - ((l & 3) << 3))) >> 24;
                int st = l / STEP;
                int kk = l - st * STEP;
                int word = b.get((nb * steps + st) * TILE * STEP / 4 + colBase + (kk >> 1) * 4);
                int bv = (word << (24 - colShift - ((kk & 1) << 3))) >> 24;
                acc += av * bv;
            }
            int z = (int) zp.get(0) - offset;
            float add = hasBias != 0 ? bias.get(col) : 0.0f;
            int out = row * n + col;
            if (hw > 0) {
                int image = row / hw;
                out = (image * n + col) * hw + row - image * hw;
            }
            c.set(out, sx.get(0) * sw.get(col) * (acc - z * colsum.get(col)) + add);
        }
    }

    /** {@link MmaInt8Kernels#gemmConv} without tensor cores: A gathered from the NHWC activations, one thread per output. */
    public static void gemmConv(KernelContext ctx, IntArray a, IntArray b, FloatArray c, int m, int n, int k, FloatArray sx, FloatArray zp, int offset, FloatArray sw,
            IntArray colsum, FloatArray bias, int hasBias, int hw, IntArray geometry) {
        int col = ctx.globalIdx;
        int row = ctx.globalIdy;
        if ((col < n) & (row < m)) {
            int cp = geometry.get(0);
            int c4s = geometry.get(1);
            int h = geometry.get(2);
            int wd = geometry.get(3);
            int oh = geometry.get(4);
            int ow = geometry.get(5);
            int kw = geometry.get(7);
            int image = row / (oh * ow);
            int pixel = row - image * oh * ow;
            int oy = pixel / ow;
            int ox = pixel - oy * ow;
            int kp = (k + STEP - 1) / STEP * STEP;
            int steps = kp / STEP;
            int nb = col / TILE;
            int jj = col - nb * TILE;
            int colBase = (jj >> 3) * 64 + ((jj & 7) >> 1);
            int colShift = (jj & 1) << 4;
            int z = (int) zp.get(0) - offset;
            int acc = 0;
            for (int l = 0; l < k; l++) {
                int tap = l / cp;
                int ci = l - tap * cp;
                int ky = tap / kw;
                int iy = oy * geometry.get(8) - geometry.get(10) + ky * geometry.get(12);
                int ix = ox * geometry.get(9) - geometry.get(11) + (tap - ky * kw) * geometry.get(13);
                int av = z;
                if ((iy >= 0) & (iy < h) & (ix >= 0) & (ix < wd)) {
                    av = (a.get(((image * h + iy) * wd + ix) * c4s + (ci >> 2)) << (24 - ((ci & 3) << 3))) >> 24;
                }
                int st = l / STEP;
                int kk = l - st * STEP;
                int word = b.get((nb * steps + st) * TILE * STEP / 4 + colBase + (kk >> 1) * 4);
                int bv = (word << (24 - colShift - ((kk & 1) << 3))) >> 24;
                acc += av * bv;
            }
            float add = hasBias != 0 ? bias.get(col) : 0.0f;
            int image2 = row / hw;
            c.set((image2 * n + col) * hw + row - image2 * hw, sx.get(0) * sw.get(col) * (acc - z * colsum.get(col)) + add);
        }
    }
}
