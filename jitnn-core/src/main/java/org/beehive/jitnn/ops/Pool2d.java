package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.PoolKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Max and average 2-D pooling over NCHW, square window, symmetric stride and padding, floor output
 * size; average counts padding (PyTorch defaults). Max pooling also outputs the argmax indices
 * used by its gradient. iattr = [N, C, H, W, OH, OW, k, stride, pad, mode (0 max, 1 avg)].
 */
public final class Pool2d implements Op {

    static final Pool2d OP = new Pool2d();
    static final Backward BACKWARD = new Backward();

    private Pool2d() {
    }

    public static Tensor max(Tensor x, int k, int stride, int pad) {
        return apply(x, k, stride, pad, 0);
    }

    public static Tensor avg(Tensor x, int k, int stride, int pad) {
        return apply(x, k, stride, pad, 1);
    }

    private static Tensor apply(Tensor x, int k, int stride, int pad, int mode) {
        Checks.f32(x);
        Checks.rank(x, 4);
        if (k < 1 || stride < 1 || pad < 0 || 2 * pad > k) {
            throw new IllegalArgumentException("pool2d: bad kernel/stride/pad " + k + "/" + stride + "/" + pad);
        }
        int n = x.dim(0);
        int c = x.dim(1);
        int h = x.dim(2);
        int w = x.dim(3);
        int oh = (h + 2 * pad - k) / stride + 1;
        int ow = (w + 2 * pad - k) / stride + 1;
        Tensor out = Tensor.result(Shape.of(n, c, oh, ow), DType.F32);
        int[] a = { n, c, h, w, oh, ow, k, stride, pad, mode };
        if (mode == 0) {
            Tensor arg = Tensor.result(out.shape(), DType.I32);
            Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out, arg }, a, null, true);
        } else {
            Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, a, null, true);
        }
        return out;
    }

    @Override
    public String name() {
        return "pool2d";
    }

    @Override
    public void emit(Emitter e, Node node) {
        int[] a = node.iattr;
        int h = a[2], w = a[3], oh = a[4], ow = a[5], k = a[6], st = a[7], pd = a[8];
        int total = node.out(0).numel();
        FloatArray x = e.f32(node.in(0));
        FloatArray out = e.f32(node.out(0));
        if (a[9] == 0) {
            IntArray arg = e.i32(node.out(1));
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, PoolKernels::maxForward, e.context(), x, out, arg, h, w, oh, ow, k, st, pd, total));
        } else {
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, PoolKernels::avgForward, e.context(), x, out, h, w, oh, ow, k, st, pd, total));
        }
    }

    @Override
    public boolean cpu(Node node) {
        int[] a = node.iattr;
        boolean max = a[9] == 0;
        return CpuKernels.pool2d(node.in(0), node.out(0), max ? node.out(1) : null, a[0] * a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], max);
    }

    @Override
    public void reference(Node node) {
        int[] a = node.iattr;
        int planes = a[0] * a[1], h = a[2], w = a[3], oh = a[4], ow = a[5], k = a[6], st = a[7], pd = a[8];
        FloatArray x = (FloatArray) node.in(0).storage().host();
        FloatArray out = (FloatArray) node.out(0).storage().host();
        IntArray arg = a[9] == 0 ? (IntArray) node.out(1).storage().host() : null;
        for (int p = 0; p < planes; p++) {
            for (int i = 0; i < oh; i++) {
                for (int j = 0; j < ow; j++) {
                    double best = Double.NEGATIVE_INFINITY;
                    int bestAt = p * h * w;
                    double sum = 0;
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int ih = i * st - pd + kh;
                            int iw = j * st - pd + kw;
                            if (ih < 0 || ih >= h || iw < 0 || iw >= w) {
                                continue;
                            }
                            int at = p * h * w + ih * w + iw;
                            double v = x.get(at);
                            sum += v;
                            if (v > best) {
                                best = v;
                                bestAt = at;
                            }
                        }
                    }
                    int o = (p * oh + i) * ow + j;
                    if (arg != null) {
                        out.set(o, (float) best);
                        arg.set(o, bestAt);
                    } else {
                        out.set(o, (float) (sum / (k * k)));
                    }
                }
            }
        }
    }

    @Override
    public Tensor[] backward(Node node, Tensor gy) {
        Tensor dx = Tensor.result(node.in(0).shape(), DType.F32);
        Tensor[] in = node.iattr[9] == 0 ? new Tensor[] { gy, node.out(1) } : new Tensor[] { gy };
        Tensor.record(BACKWARD, in, new Tensor[] { dx }, node.iattr.clone(), null, true);
        return new Tensor[] { dx };
    }

    /** Inputs [dy, argmax] (max) or [dy] (avg); output dx. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "pool2d_bwd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            int h = a[2], w = a[3], oh = a[4], ow = a[5], k = a[6], st = a[7], pd = a[8];
            int total = node.out(0).numel();
            FloatArray dy = e.f32(node.in(0));
            FloatArray dx = e.f32(node.out(0));
            if (a[9] == 0) {
                IntArray arg = e.i32(node.in(1));
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, PoolKernels::maxBackward, e.context(), dy, arg, dx, h, w, oh, ow, k, st, pd, total));
            } else {
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, PoolKernels::avgBackward, e.context(), dy, dx, h, w, oh, ow, k, st, pd, total));
            }
        }

        @Override
        public void reference(Node node) {
            int[] a = node.iattr;
            int planes = a[0] * a[1], h = a[2], w = a[3], oh = a[4], ow = a[5], k = a[6], st = a[7], pd = a[8];
            FloatArray dy = (FloatArray) node.in(0).storage().host();
            FloatArray dx = (FloatArray) node.out(0).storage().host();
            double[] acc = new double[node.out(0).numel()];
            if (a[9] == 0) {
                IntArray arg = (IntArray) node.in(1).storage().host();
                for (int o = 0; o < node.in(0).numel(); o++) {
                    acc[arg.get(o)] += dy.get(o);
                }
            } else {
                for (int p = 0; p < planes; p++) {
                    for (int i = 0; i < oh; i++) {
                        for (int j = 0; j < ow; j++) {
                            double g = dy.get((p * oh + i) * ow + j) / (k * k);
                            for (int kh = 0; kh < k; kh++) {
                                for (int kw = 0; kw < k; kw++) {
                                    int ih = i * st - pd + kh;
                                    int iw = j * st - pd + kw;
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                        acc[p * h * w + ih * w + iw] += g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < acc.length; i++) {
                dx.set(i, (float) acc[i]);
            }
        }
    }
}
