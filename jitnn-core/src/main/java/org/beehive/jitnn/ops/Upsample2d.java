package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.ResizeKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Nearest or bilinear resizing of x [N, C, H, W] to [N, C, OH, OW], PyTorch's
 * {@code F.interpolate}. iattr = [planes, H, W, OH, OW, mode, alignCorners]; fattr = [rh, rw],
 * the source steps per output pixel (see {@link ResizeKernels}).
 */
public final class Upsample2d implements Op {

    static final Upsample2d OP = new Upsample2d();
    static final Backward BACKWARD = new Backward();

    private static final int P = 0, H = 1, W = 2, OH = 3, OW = 4, MODE = 5, ALIGN = 6;

    private Upsample2d() {
    }

    /** By scale factor (output size floor(H * scale)); the source step is 1 / scale, as PyTorch. */
    public static Tensor scale(Tensor x, float scaleH, float scaleW, int mode, boolean alignCorners) {
        Checks.rank(x, 4);
        int oh = (int) Math.floor(x.dim(2) * (double) scaleH);
        int ow = (int) Math.floor(x.dim(3) * (double) scaleW);
        float rh = alignCorners ? cornerStep(x.dim(2), oh) : 1.0f / scaleH;
        float rw = alignCorners ? cornerStep(x.dim(3), ow) : 1.0f / scaleW;
        return apply(x, oh, ow, rh, rw, mode, alignCorners);
    }

    /** To a given size; the source step is in / out (or (in - 1) / (out - 1) with align_corners). */
    public static Tensor size(Tensor x, int oh, int ow, int mode, boolean alignCorners) {
        Checks.rank(x, 4);
        float rh = alignCorners ? cornerStep(x.dim(2), oh) : (float) x.dim(2) / oh;
        float rw = alignCorners ? cornerStep(x.dim(3), ow) : (float) x.dim(3) / ow;
        return apply(x, oh, ow, rh, rw, mode, alignCorners);
    }

    private static float cornerStep(int in, int out) {
        return out > 1 ? (float) (in - 1) / (out - 1) : 0.0f;
    }

    private static Tensor apply(Tensor x, int oh, int ow, float rh, float rw, int mode, boolean align) {
        Checks.f32(x);
        if (oh < 1 || ow < 1) {
            throw new IllegalArgumentException("upsample: empty output " + oh + "x" + ow + " for " + x.shape());
        }
        if (mode == ResizeKernels.NEAREST && align) {
            throw new IllegalArgumentException("upsample: align_corners applies to bilinear only");
        }
        int[] a = { x.dim(0) * x.dim(1), x.dim(2), x.dim(3), oh, ow, mode, align ? 1 : 0 };
        Tensor out = Tensor.result(Shape.of(x.dim(0), x.dim(1), oh, ow), DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, a, new float[] { rh, rw }, true);
        return out;
    }

    @Override
    public String name() {
        return "upsample2d";
    }

    @Override
    public void emit(Emitter e, Node node) {
        int[] a = node.iattr;
        FloatArray x = e.f32(node.in(0));
        FloatArray y = e.f32(node.out(0));
        int h = a[H], w = a[W], oh = a[OH], ow = a[OW], mode = a[MODE], align = a[ALIGN];
        float rh = node.fattr[0], rw = node.fattr[1];
        int total = a[P] * oh * ow;
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ResizeKernels::forward, e.context(), x, y, h, w, oh, ow, rh, rw, mode, align, total));
    }

    @Override
    public void reference(Node node) {
        int[] a = node.iattr;
        FloatArray x = (FloatArray) node.in(0).storage().host();
        FloatArray y = (FloatArray) node.out(0).storage().host();
        double[][] wy = weights(a[H], a[OH], node.fattr[0], a[MODE], a[ALIGN]);
        double[][] wx = weights(a[W], a[OW], node.fattr[1], a[MODE], a[ALIGN]);
        for (int p = 0; p < a[P]; p++) {
            for (int oy = 0; oy < a[OH]; oy++) {
                for (int ox = 0; ox < a[OW]; ox++) {
                    double acc = 0;
                    for (int iy = 0; iy < a[H]; iy++) {
                        if (wy[oy][iy] == 0) {
                            continue;
                        }
                        for (int ix = 0; ix < a[W]; ix++) {
                            acc += wy[oy][iy] * wx[ox][ix] * x.get((p * a[H] + iy) * a[W] + ix);
                        }
                    }
                    y.set((p * a[OH] + oy) * a[OW] + ox, (float) acc);
                }
            }
        }
    }

    /**
     * The interpolation matrix along one axis, [out][in], written independently of the kernels:
     * PyTorch's source index rules in double precision.
     */
    static double[][] weights(int in, int out, float r, int mode, int align) {
        double[][] m = new double[out][in];
        for (int o = 0; o < out; o++) {
            if (mode == ResizeKernels.NEAREST) {
                m[o][Math.min((int) Math.floor(o * (double) r), in - 1)] = 1;
            } else {
                double s = align != 0 ? r * (double) o : Math.max(r * (o + 0.5) - 0.5, 0);
                int i0 = (int) Math.floor(s);
                int i1 = Math.min(i0 + 1, in - 1);
                double l1 = s - i0;
                m[o][i0] += 1 - l1;
                m[o][i1] += l1;
            }
        }
        return m;
    }

    @Override
    public Tensor[] backward(Node node, Tensor gy) {
        Tensor gx = Tensor.result(node.in(0).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { gy }, new Tensor[] { gx }, node.iattr.clone(), node.fattr.clone(), true);
        return new Tensor[] { gx };
    }

    /** dx: each input pixel gathers the output gradients that read it. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "upsample2d_bwd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            FloatArray gy = e.f32(node.in(0));
            FloatArray gx = e.f32(node.out(0));
            int h = a[H], w = a[W], oh = a[OH], ow = a[OW], mode = a[MODE], align = a[ALIGN];
            float rh = node.fattr[0], rw = node.fattr[1];
            int total = a[P] * h * w;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ResizeKernels::backward, e.context(), gy, gx, h, w, oh, ow, rh, rw, mode, align, total));
        }

        @Override
        public void reference(Node node) {
            int[] a = node.iattr;
            FloatArray gy = (FloatArray) node.in(0).storage().host();
            FloatArray gx = (FloatArray) node.out(0).storage().host();
            double[][] wy = weights(a[H], a[OH], node.fattr[0], a[MODE], a[ALIGN]);
            double[][] wx = weights(a[W], a[OW], node.fattr[1], a[MODE], a[ALIGN]);
            for (int p = 0; p < a[P]; p++) {
                for (int iy = 0; iy < a[H]; iy++) {
                    for (int ix = 0; ix < a[W]; ix++) {
                        double acc = 0;
                        for (int oy = 0; oy < a[OH]; oy++) {
                            if (wy[oy][iy] == 0) {
                                continue;
                            }
                            for (int ox = 0; ox < a[OW]; ox++) {
                                acc += wy[oy][iy] * wx[ox][ix] * gy.get((p * a[OH] + oy) * a[OW] + ox);
                            }
                        }
                        gx.set((p * a[H] + iy) * a[W] + ix, (float) acc);
                    }
                }
            }
        }
    }
}
