package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * N-dimensional spatial kernels over [N, C, D, H, W] (1-D and 2-D data as D = H = 1 or D = 1):
 * unfolding windows into columns (im2col, for Conv1D/Conv3D and 1-D/3-D pooling) and its
 * gathering gradient, adaptive average/max pooling, and constant/reflect/replicate padding. Every
 * gradient gathers per input element, so nothing is atomic and results are deterministic.
 *
 * <p>Unfold meta: [C, D, H, W, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW]; the
 * columns are [N, C * KD * KH * KW, OD * OH * OW].
 */
public final class SpatialKernels {

    public static final int CONSTANT = 0;
    public static final int REFLECT = 1;
    public static final int REPLICATE = 2;

    private SpatialKernels() {
    }

    /** One thread per column element: the input value its window position reads, or 0 in the padding. */
    public static void unfold(KernelContext kc, FloatArray x, FloatArray cols, IntArray m, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int c = m.get(0);
            int d = m.get(1);
            int h = m.get(2);
            int w = m.get(3);
            int kd = m.get(4);
            int kh = m.get(5);
            int kw = m.get(6);
            int od = m.get(16);
            int oh = m.get(17);
            int ow = m.get(18);
            int l = od * oh * ow;
            int pos = i % l;
            int row = (i / l) % (c * kd * kh * kw);
            int n = i / (l * c * kd * kh * kw);
            int ox = pos % ow;
            int oy = (pos / ow) % oh;
            int oz = pos / (ow * oh);
            int jx = row % kw;
            int jy = (row / kw) % kh;
            int jz = (row / (kw * kh)) % kd;
            int ch = row / (kw * kh * kd);
            int iz = oz * m.get(7) - m.get(10) + jz * m.get(13);
            int iy = oy * m.get(8) - m.get(11) + jy * m.get(14);
            int ix = ox * m.get(9) - m.get(12) + jx * m.get(15);
            float v = 0.0f;
            if ((iz >= 0) & (iz < d) & (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)) {
                v = x.get((((n * c + ch) * d + iz) * h + iy) * w + ix);
            }
            cols.set(i, v);
        }
    }

    /** Gradient of {@link #unfold}: each input element sums the column entries that read it. */
    public static void fold(KernelContext kc, FloatArray gcols, FloatArray gx, IntArray m, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int c = m.get(0);
            int d = m.get(1);
            int h = m.get(2);
            int w = m.get(3);
            int kd = m.get(4);
            int kh = m.get(5);
            int kw = m.get(6);
            int od = m.get(16);
            int oh = m.get(17);
            int ow = m.get(18);
            int l = od * oh * ow;
            int ix = i % w;
            int iy = (i / w) % h;
            int iz = (i / (w * h)) % d;
            int ch = (i / (w * h * d)) % c;
            int n = i / (w * h * d * c);
            float acc = 0.0f;
            for (int jz = 0; jz < kd; jz++) {
                int tz = iz + m.get(10) - jz * m.get(13);
                int oz = tz / m.get(7);
                if ((tz >= 0) & (tz - oz * m.get(7) == 0) & (oz < od)) {
                    for (int jy = 0; jy < kh; jy++) {
                        int ty = iy + m.get(11) - jy * m.get(14);
                        int oy = ty / m.get(8);
                        if ((ty >= 0) & (ty - oy * m.get(8) == 0) & (oy < oh)) {
                            for (int jx = 0; jx < kw; jx++) {
                                int tx = ix + m.get(12) - jx * m.get(15);
                                int ox = tx / m.get(9);
                                if ((tx >= 0) & (tx - ox * m.get(9) == 0) & (ox < ow)) {
                                    int row = ((ch * kd + jz) * kh + jy) * kw + jx;
                                    acc += gcols.get((n * c * kd * kh * kw + row) * l + (oz * oh + oy) * ow + ox);
                                }
                            }
                        }
                    }
                }
            }
            gx.set(i, acc);
        }
    }

    /** First input index of adaptive bin o of out over in: floor(o in / out). */
    public static int binStart(int o, int in, int out) {
        return o * in / out;
    }

    /** One past the last input index of adaptive bin o: ceil((o + 1) in / out). */
    public static int binEnd(int o, int in, int out) {
        return ((o + 1) * in + out - 1) / out;
    }

    /**
     * Adaptive average (mode 0) or max (mode 1) pooling of [P, D, H, W] to [P, OD, OH, OW] (PyTorch's
     * bins). Max pooling also writes the flat input index of each maximum to {@code arg}.
     * meta = [D, H, W, OD, OH, OW].
     */
    public static void adaptivePool(KernelContext kc, FloatArray x, FloatArray y, IntArray arg, IntArray m, int mode, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int d = m.get(0);
            int h = m.get(1);
            int w = m.get(2);
            int od = m.get(3);
            int oh = m.get(4);
            int ow = m.get(5);
            int ox = i % ow;
            int oy = (i / ow) % oh;
            int oz = (i / (ow * oh)) % od;
            int p = i / (ow * oh * od);
            int z0 = binStart(oz, d, od);
            int z1 = binEnd(oz, d, od);
            int y0 = binStart(oy, h, oh);
            int y1 = binEnd(oy, h, oh);
            int x0 = binStart(ox, w, ow);
            int x1 = binEnd(ox, w, ow);
            float acc = 0.0f;
            float best = -3.4028235e38f;
            int at = 0;
            for (int z = z0; z < z1; z++) {
                for (int yy = y0; yy < y1; yy++) {
                    for (int xx = x0; xx < x1; xx++) {
                        int idx = ((p * d + z) * h + yy) * w + xx;
                        float v = x.get(idx);
                        acc += v;
                        if (v > best) {
                            best = v;
                            at = idx;
                        }
                    }
                }
            }
            if (mode == 0) {
                y.set(i, acc / ((z1 - z0) * (y1 - y0) * (x1 - x0)));
            } else {
                y.set(i, best);
                arg.set(i, at);
            }
        }
    }

    /** Gradient of {@link #adaptivePool}: each input gathers from the bins that contain it. */
    public static void adaptivePoolBackward(KernelContext kc, FloatArray gy, IntArray arg, FloatArray gx, IntArray m, int mode, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int d = m.get(0);
            int h = m.get(1);
            int w = m.get(2);
            int od = m.get(3);
            int oh = m.get(4);
            int ow = m.get(5);
            int ix = i % w;
            int iy = (i / w) % h;
            int iz = (i / (w * h)) % d;
            int p = i / (w * h * d);
            float acc = 0.0f;
            for (int oz = 0; oz < od; oz++) {
                int z0 = binStart(oz, d, od);
                int z1 = binEnd(oz, d, od);
                if ((iz >= z0) & (iz < z1)) {
                    for (int oy = 0; oy < oh; oy++) {
                        int y0 = binStart(oy, h, oh);
                        int y1 = binEnd(oy, h, oh);
                        if ((iy >= y0) & (iy < y1)) {
                            for (int ox = 0; ox < ow; ox++) {
                                int x0 = binStart(ox, w, ow);
                                int x1 = binEnd(ox, w, ow);
                                if ((ix >= x0) & (ix < x1)) {
                                    int o = ((p * od + oz) * oh + oy) * ow + ox;
                                    if (mode == 0) {
                                        acc += gy.get(o) / ((z1 - z0) * (y1 - y0) * (x1 - x0));
                                    } else if (arg.get(o) == i) {
                                        acc += gy.get(o);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            gx.set(i, acc);
        }
    }

    /** Source coordinate of output o along an axis of size in padded by before (or -1 for a constant pad). */
    public static int padSource(int o, int before, int in, int mode) {
        int i = o - before;
        int r = i;
        if ((i < 0) | (i >= in)) {
            if (mode == REFLECT) {
                int period = 2 * in - 2;
                int t = i < 0 ? -i : i;
                t = period > 0 ? t % period : 0;
                r = t < in ? t : period - t;
            } else if (mode == REPLICATE) {
                r = i < 0 ? 0 : in - 1;
            } else {
                r = -1;
            }
        }
        return r;
    }

    /**
     * Padding of [P, D, H, W] to [P, OD, OH, OW]: constant {@code value}, reflection without the edge
     * (PyTorch's reflect) or edge replication. meta = [D, H, W, OD, OH, OW, before D, before H, before W].
     */
    public static void pad(KernelContext kc, FloatArray x, FloatArray y, IntArray m, int mode, float value, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int d = m.get(0);
            int h = m.get(1);
            int w = m.get(2);
            int od = m.get(3);
            int oh = m.get(4);
            int ow = m.get(5);
            int ox = i % ow;
            int oy = (i / ow) % oh;
            int oz = (i / (ow * oh)) % od;
            int p = i / (ow * oh * od);
            int sz = padSource(oz, m.get(6), d, mode);
            int sy = padSource(oy, m.get(7), h, mode);
            int sx = padSource(ox, m.get(8), w, mode);
            float v = value;
            if ((sz >= 0) & (sy >= 0) & (sx >= 0)) {
                v = x.get(((p * d + sz) * h + sy) * w + sx);
            }
            y.set(i, v);
        }
    }

    /** Gradient of {@link #pad}: each input sums the outputs that copy it (several, for reflect and replicate). */
    public static void padBackward(KernelContext kc, FloatArray gy, FloatArray gx, IntArray m, int mode, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int d = m.get(0);
            int h = m.get(1);
            int w = m.get(2);
            int od = m.get(3);
            int oh = m.get(4);
            int ow = m.get(5);
            int ix = i % w;
            int iy = (i / w) % h;
            int iz = (i / (w * h)) % d;
            int p = i / (w * h * d);
            float acc = 0.0f;
            for (int oz = 0; oz < od; oz++) {
                if (padSource(oz, m.get(6), d, mode) == iz) {
                    for (int oy = 0; oy < oh; oy++) {
                        if (padSource(oy, m.get(7), h, mode) == iy) {
                            for (int ox = 0; ox < ow; ox++) {
                                if (padSource(ox, m.get(8), w, mode) == ix) {
                                    acc += gy.get(((p * od + oz) * oh + oy) * ow + ox);
                                }
                            }
                        }
                    }
                }
            }
            gx.set(i, acc);
        }
    }
}
