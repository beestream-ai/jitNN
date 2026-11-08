package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Nearest and bilinear resizing of [P, H, W] planes to [P, OH, OW], PyTorch's interpolate
 * semantics. {@code rh}, {@code rw} are the source steps per output pixel: 1 / scale_factor, or
 * in / out for a requested size, or (in - 1) / (out - 1) for bilinear with align_corners. The
 * backward kernels gather, one thread per input pixel, over the outputs that read it, so they
 * are deterministic.
 */
public final class ResizeKernels {

    public static final int NEAREST = 0;
    public static final int BILINEAR = 1;

    private ResizeKernels() {
    }

    /** Nearest source index of output {@code o}. */
    public static int nearest(int o, float r, int in) {
        int i = (int) (o * r);
        return i < in - 1 ? i : in - 1;
    }

    /** Bilinear source coordinate of output {@code o}, clamped at 0 unless align_corners. */
    public static float source(int o, float r, int align) {
        float s;
        if (align != 0) {
            s = r * o;
        } else {
            s = r * (o + 0.5f) - 0.5f;
            s = s > 0.0f ? s : 0.0f;
        }
        return s;
    }

    /** Weight of input {@code i} in output {@code o} along one axis (bilinear). */
    public static float weight(int o, int i, float r, int in, int align) {
        float s = source(o, r, align);
        int i0 = (int) s;
        int i1 = i0 < in - 1 ? i0 + 1 : i0;
        float l1 = s - i0;
        float w = 0.0f;
        if (i0 == i) {
            w += 1.0f - l1;
        }
        if (i1 == i) {
            w += l1;
        }
        return w;
    }

    public static void forward(KernelContext kc, FloatArray x, FloatArray y, int h, int w, int oh, int ow, float rh, float rw, int mode, int align, int total) {
        int idx = kc.globalIdx;
        if (idx < total) {
            int ox = idx % ow;
            int oy = (idx / ow) % oh;
            int p = idx / (ow * oh);
            int base = p * h * w;
            float v;
            if (mode == NEAREST) {
                v = x.get(base + nearest(oy, rh, h) * w + nearest(ox, rw, w));
            } else {
                float sy = source(oy, rh, align);
                float sx = source(ox, rw, align);
                int y0 = (int) sy;
                int x0 = (int) sx;
                int y1 = y0 < h - 1 ? y0 + 1 : y0;
                int x1 = x0 < w - 1 ? x0 + 1 : x0;
                float ly = sy - y0;
                float lx = sx - x0;
                float top = (1.0f - lx) * x.get(base + y0 * w + x0) + lx * x.get(base + y0 * w + x1);
                float bottom = (1.0f - lx) * x.get(base + y1 * w + x0) + lx * x.get(base + y1 * w + x1);
                v = (1.0f - ly) * top + ly * bottom;
            }
            y.set(idx, v);
        }
    }

    /** First output (inclusive) that can read input {@code i}: a margin around i / r. */
    public static int first(int i, float r) {
        int o = r > 0.0f ? (int) ((i - 1) / r) - 2 : 0;
        return o > 0 ? o : 0;
    }

    /** Last output (exclusive) that can read input {@code i}. */
    public static int last(int i, float r, int out) {
        int o = r > 0.0f ? (int) ((i + 2) / r) + 3 : out;
        return o < out ? o : out;
    }

    public static void backward(KernelContext kc, FloatArray gy, FloatArray gx, int h, int w, int oh, int ow, float rh, float rw, int mode, int align, int total) {
        int idx = kc.globalIdx;
        if (idx < total) {
            int ix = idx % w;
            int iy = (idx / w) % h;
            int p = idx / (w * h);
            int base = p * oh * ow;
            int oy0 = first(iy, rh);
            int oy1 = last(iy, rh, oh);
            int ox0 = first(ix, rw);
            int ox1 = last(ix, rw, ow);
            float acc = 0.0f;
            for (int oy = oy0; oy < oy1; oy++) {
                float wy;
                if (mode == NEAREST) {
                    wy = nearest(oy, rh, h) == iy ? 1.0f : 0.0f;
                } else {
                    wy = weight(oy, iy, rh, h, align);
                }
                if (wy != 0.0f) {
                    float row = 0.0f;
                    for (int ox = ox0; ox < ox1; ox++) {
                        float wx;
                        if (mode == NEAREST) {
                            wx = nearest(ox, rw, w) == ix ? 1.0f : 0.0f;
                        } else {
                            wx = weight(ox, ix, rw, w, align);
                        }
                        row += wx * gy.get(base + oy * ow + ox);
                    }
                    acc += wy * row;
                }
            }
            gx.set(idx, acc);
        }
    }
}
