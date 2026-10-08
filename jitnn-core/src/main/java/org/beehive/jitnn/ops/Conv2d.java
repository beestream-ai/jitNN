package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.ConvKernels;
import org.beehive.jitnn.kernels.GemmKernels;
import org.beehive.jitnn.kernels.ReduceKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * 2-D convolution, PyTorch semantics and layout: x [N, Cin, H, W], weight [Cout, Cin/groups, K, K],
 * optional bias [Cout], square kernel, symmetric stride, padding and dilation. Bias and ReLU can be
 * fused into the output. iattr = [N, Cin, H, W, Cout, OH, OW, K, stride, pad, dil, groups,
 * hasBias, relu].
 */
public final class Conv2d implements Op {

    static final Conv2d OP = new Conv2d();
    static final BackwardData BACKWARD_DATA = new BackwardData();
    static final BackwardWeight BACKWARD_WEIGHT = new BackwardWeight();

    private static final int N = 0, CIN = 1, H = 2, W = 3, COUT = 4, OH = 5, OW = 6, K = 7, STRIDE = 8, PAD = 9, DIL = 10, GROUPS = 11, BIAS = 12, RELU = 13;

    private Conv2d() {
    }

    public static Tensor apply(Tensor x, Tensor w, Tensor bias, int stride, int pad, int dil, int groups, boolean relu) {
        Checks.f32(x, w);
        Checks.rank(x, 4);
        Checks.rank(w, 4);
        int n = x.dim(0);
        int cin = x.dim(1);
        int h = x.dim(2);
        int wd = x.dim(3);
        int cout = w.dim(0);
        int k = w.dim(2);
        if (w.dim(3) != k) {
            throw new IllegalArgumentException("conv2d: only square kernels are supported, got " + w.shape());
        }
        if (groups < 1 || cin % groups != 0 || cout % groups != 0 || w.dim(1) != cin / groups) {
            throw new IllegalArgumentException("conv2d: input " + x.shape() + ", weight " + w.shape() + " and groups " + groups + " do not match");
        }
        if (stride < 1 || pad < 0 || dil < 1) {
            throw new IllegalArgumentException("conv2d: bad stride/pad/dilation " + stride + "/" + pad + "/" + dil);
        }
        int oh = (h + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        int ow = (wd + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        if (oh < 1 || ow < 1) {
            throw new IllegalArgumentException("conv2d: output would be empty for input " + x.shape());
        }
        if (bias != null) {
            Checks.f32(bias);
            if (bias.numel() != cout) {
                throw new IllegalArgumentException("conv2d: bias " + bias.shape() + " for " + cout + " channels");
            }
        }
        Tensor out = Tensor.result(Shape.of(n, cout, oh, ow), DType.F32);
        int[] a = { n, cin, h, wd, cout, oh, ow, k, stride, pad, dil, groups, bias == null ? 0 : 1, relu ? 1 : 0 };
        Tensor[] in = bias == null ? new Tensor[] { x, w } : new Tensor[] { x, w, bias };
        Tensor.record(OP, in, new Tensor[] { out }, a, null, true);
        return out;
    }

    static boolean depthwise(int[] a) {
        return a[GROUPS] == a[CIN] && a[CIN] == a[COUT];
    }

    /** The padded, prefetching implicit-GEMM kernels unless {@code -Djitnn.conv=simple}. */
    static final boolean PADDED = !"simple".equals(System.getProperty("jitnn.conv"));

    @Override
    public String name() {
        return "conv2d";
    }

    @Override
    public void emit(Emitter e, Node node) {
        int[] a = node.iattr;
        FloatArray x = e.f32(node.in(0));
        FloatArray w = e.f32(node.in(1));
        FloatArray out = e.f32(node.out(0));
        FloatArray bias = a[BIAS] != 0 ? e.f32(node.in(2)) : e.dummyF32();
        int epilogue = (a[BIAS] != 0 ? GemmKernels.BIAS : 0) | (a[RELU] != 0 ? GemmKernels.RELU : 0);
        int cin = a[CIN], h = a[H], wd = a[W], cout = a[COUT], oh = a[OH], ow = a[OW], k = a[K], st = a[STRIDE], pd = a[PAD], dl = a[DIL], gr = a[GROUPS];
        if (depthwise(a)) {
            int total = a[N] * cout * oh * ow;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ConvKernels::depthwiseForward, e.context(), x, w, out, bias, cout, h, wd, oh, ow, k, st, pd, dl, total, epilogue));
            return;
        }
        int m = a[N] * oh * ow;
        int tilesN = Emitter.ceilDiv(cout / gr, 64);
        int tilesM = Emitter.ceilDiv(m, 64);
        int kg = (cin / gr) * k * k;
        int slices = MatMul.forwardSlices(tilesN * gr * tilesM, kg);
        if (slices > 1) {
            // too few output tiles to fill the GPU: split the (ci, kh, kw) reduction, then sum with the epilogue
            int chunk = Emitter.ceilDiv(Emitter.ceilDiv(kg, slices), 16) * 16;
            int sl = Emitter.ceilDiv(kg, chunk);
            int total = m * cout;
            FloatArray part = e.scratch(sl * total);
            e.task(Emitter.blocks2D(tilesN * gr, tilesM * sl, 16, 16),
                    (g, id) -> {
                        if (PADDED) {
                            g.task(id, ConvKernels::conv2dForwardSplitKPd, e.context(), x, w, part, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, m, tilesN, tilesM, chunk);
                        } else {
                            g.task(id, ConvKernels::conv2dForwardSplitK, e.context(), x, w, part, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, m, tilesN, tilesM, chunk);
                        }
                    });
            int plane = oh * ow;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ConvKernels::sumSplitK, e.context(), part, out, bias, sl, total, cout, plane, epilogue));
            return;
        }
        e.task(Emitter.blocks2D(tilesN * gr, tilesM, 16, 16),
                (g, id) -> {
                        if (PADDED) {
                            g.task(id, ConvKernels::conv2dForwardPd, e.context(), x, w, out, bias, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, m, epilogue, tilesN);
                        } else {
                            g.task(id, ConvKernels::conv2dForward, e.context(), x, w, out, bias, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, m, epilogue, tilesN);
                        }
                    });
    }

    @Override
    public void reference(Node node) {
        int[] a = node.iattr;
        FloatArray x = (FloatArray) node.in(0).storage().host();
        FloatArray w = (FloatArray) node.in(1).storage().host();
        FloatArray b = a[BIAS] != 0 ? (FloatArray) node.in(2).storage().host() : null;
        FloatArray out = (FloatArray) node.out(0).storage().host();
        int cg = a[CIN] / a[GROUPS];
        int cog = a[COUT] / a[GROUPS];
        int k = a[K];
        for (int n = 0; n < a[N]; n++) {
            for (int co = 0; co < a[COUT]; co++) {
                int g = co / cog;
                for (int oh = 0; oh < a[OH]; oh++) {
                    for (int ow = 0; ow < a[OW]; ow++) {
                        double acc = b == null ? 0 : b.get(co);
                        for (int ci = 0; ci < cg; ci++) {
                            for (int kh = 0; kh < k; kh++) {
                                int ih = oh * a[STRIDE] - a[PAD] + kh * a[DIL];
                                if (ih < 0 || ih >= a[H]) {
                                    continue;
                                }
                                for (int kw = 0; kw < k; kw++) {
                                    int iw = ow * a[STRIDE] - a[PAD] + kw * a[DIL];
                                    if (iw < 0 || iw >= a[W]) {
                                        continue;
                                    }
                                    acc += (double) x.get(((n * a[CIN] + g * cg + ci) * a[H] + ih) * a[W] + iw) * w.get(((co * cg + ci) * k + kh) * k + kw);
                                }
                            }
                        }
                        out.set(((n * a[COUT] + co) * a[OH] + oh) * a[OW] + ow, (float) (a[RELU] != 0 ? Math.max(acc, 0) : acc));
                    }
                }
            }
        }
    }

    @Override
    public Tensor[] backward(Node node, Tensor gy) {
        int[] a = node.iattr;
        Tensor x = node.in(0);
        Tensor w = node.in(1);
        Tensor g = a[RELU] != 0 ? Binary.apply(Binary.Kind.RELU_GRAD, node.out(0), gy) : gy;
        Tensor gx = null;
        Tensor gw = null;
        if (x.requiresGrad()) {
            gx = Tensor.result(x.shape(), DType.F32);
            Tensor.record(BACKWARD_DATA, new Tensor[] { g, w }, new Tensor[] { gx }, a.clone(), null, true);
        }
        if (w.requiresGrad()) {
            gw = Tensor.result(w.shape(), DType.F32);
            Tensor.record(BACKWARD_WEIGHT, new Tensor[] { g, x }, new Tensor[] { gw }, a.clone(), null, true);
        }
        if (a[BIAS] == 0) {
            return new Tensor[] { gx, gw };
        }
        Tensor b = node.in(2);
        Tensor gb = b.requiresGrad() ? g.reshape(a[N], a[COUT], a[OH] * a[OW]).sum(2, false).sum(0, false) : null;
        return new Tensor[] { gx, gw, gb };
    }

    /** dx from dy and the weight. */
    static final class BackwardData implements Op {

        @Override
        public String name() {
            return "conv2d_bwd_data";
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            FloatArray dy = e.f32(node.in(0));
            FloatArray w = e.f32(node.in(1));
            FloatArray dx = e.f32(node.out(0));
            int cin = a[CIN], h = a[H], wd = a[W], cout = a[COUT], oh = a[OH], ow = a[OW], k = a[K], st = a[STRIDE], pd = a[PAD], dl = a[DIL], gr = a[GROUPS];
            if (depthwise(a)) {
                int total = a[N] * cin * h * wd;
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ConvKernels::depthwiseBackwardData, e.context(), dy, w, dx, cin, h, wd, oh, ow, k, st, pd, dl, total));
                return;
            }
            int m = a[N] * h * wd;
            int tilesN = Emitter.ceilDiv(cin / gr, 64);
            e.task(Emitter.blocks2D(tilesN * gr, Emitter.ceilDiv(m, 64), 16, 16),
                    (g, id) -> {
                        if (PADDED) {
                            g.task(id, ConvKernels::conv2dBackwardDataPd, e.context(), dy, w, dx, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, m, tilesN);
                        } else {
                            g.task(id, ConvKernels::conv2dBackwardData, e.context(), dy, w, dx, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, m, tilesN);
                        }
                    });
        }

        @Override
        public void reference(Node node) {
            int[] a = node.iattr;
            FloatArray dy = (FloatArray) node.in(0).storage().host();
            FloatArray w = (FloatArray) node.in(1).storage().host();
            FloatArray dx = (FloatArray) node.out(0).storage().host();
            int cg = a[CIN] / a[GROUPS];
            int cog = a[COUT] / a[GROUPS];
            int k = a[K];
            double[] acc = new double[node.out(0).numel()];
            for (int n = 0; n < a[N]; n++) {
                for (int co = 0; co < a[COUT]; co++) {
                    int g = co / cog;
                    for (int oh = 0; oh < a[OH]; oh++) {
                        for (int ow = 0; ow < a[OW]; ow++) {
                            double gyv = dy.get(((n * a[COUT] + co) * a[OH] + oh) * a[OW] + ow);
                            for (int ci = 0; ci < cg; ci++) {
                                for (int kh = 0; kh < k; kh++) {
                                    int ih = oh * a[STRIDE] - a[PAD] + kh * a[DIL];
                                    if (ih < 0 || ih >= a[H]) {
                                        continue;
                                    }
                                    for (int kw = 0; kw < k; kw++) {
                                        int iw = ow * a[STRIDE] - a[PAD] + kw * a[DIL];
                                        if (iw < 0 || iw >= a[W]) {
                                            continue;
                                        }
                                        acc[((n * a[CIN] + g * cg + ci) * a[H] + ih) * a[W] + iw] += gyv * w.get(((co * cg + ci) * k + kh) * k + kw);
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

    /** dW from dy and the input; split-K over (n, oh, ow) when there are few output tiles. */
    static final class BackwardWeight implements Op {

        @Override
        public String name() {
            return "conv2d_bwd_weight";
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            FloatArray dy = e.f32(node.in(0));
            FloatArray x = e.f32(node.in(1));
            FloatArray dw = e.f32(node.out(0));
            int cin = a[CIN], h = a[H], wd = a[W], cout = a[COUT], oh = a[OH], ow = a[OW], k = a[K], st = a[STRIDE], pd = a[PAD], dl = a[DIL], gr = a[GROUPS];
            int p = a[N] * oh * ow;
            int weights = node.out(0).numel();
            if (depthwise(a)) {
                int splits = Math.max(1, Math.min(Emitter.ceilDiv(1 << 16, weights), p / 64));
                int chunk = Emitter.ceilDiv(p, splits);
                int sp = Emitter.ceilDiv(p, chunk);
                FloatArray partial = sp > 1 ? e.scratch(weights * sp) : dw;
                e.task(Emitter.grid1D(weights * sp), (g, id) -> g.task(id, ConvKernels::depthwiseBackwardWeight, e.context(), dy, x, partial, cin, h, wd, oh, ow, k, st, pd, dl, p, chunk, sp));
                if (sp > 1) {
                    e.task(Emitter.grid1D(weights), (g, id) -> g.task(id, ReduceKernels::reduceThread, e.context(), partial, dw, 1, sp, weights, ReduceKernels.SUM, 1.0f));
                }
                return;
            }
            int cog = cout / gr;
            int kg = (cin / gr) * k * k;
            int tilesN = Emitter.ceilDiv(kg, 64);
            int tilesM = Emitter.ceilDiv(cog, 64);
            int blocks = tilesN * tilesM * gr;
            int splits = Math.max(1, Math.min(Emitter.ceilDiv(256, blocks), p / 256));
            int chunk = Emitter.ceilDiv(Emitter.ceilDiv(p, splits), 16) * 16;
            int sp = Emitter.ceilDiv(p, chunk);
            FloatArray partial = sp > 1 ? e.scratch(weights * sp) : dw;
            e.task(Emitter.blocks2D(tilesN * gr, tilesM * sp, 16, 16),
                    (g, id) -> {
                        if (PADDED) {
                            g.task(id, ConvKernels::conv2dBackwardWeightPd, e.context(), dy, x, partial, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, p, tilesN, tilesM, chunk);
                        } else {
                            g.task(id, ConvKernels::conv2dBackwardWeight, e.context(), dy, x, partial, cin, h, wd, cout, oh, ow, k, st, pd, dl, gr, p, tilesN, tilesM, chunk);
                        }
                    });
            if (sp > 1) {
                e.task(Emitter.grid1D(weights), (g, id) -> g.task(id, ReduceKernels::reduceThread, e.context(), partial, dw, 1, sp, weights, ReduceKernels.SUM, 1.0f));
            }
        }

        @Override
        public void reference(Node node) {
            int[] a = node.iattr;
            FloatArray dy = (FloatArray) node.in(0).storage().host();
            FloatArray x = (FloatArray) node.in(1).storage().host();
            FloatArray dw = (FloatArray) node.out(0).storage().host();
            int cg = a[CIN] / a[GROUPS];
            int cog = a[COUT] / a[GROUPS];
            int k = a[K];
            for (int co = 0; co < a[COUT]; co++) {
                int g = co / cog;
                for (int ci = 0; ci < cg; ci++) {
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            double acc = 0;
                            for (int n = 0; n < a[N]; n++) {
                                for (int oh = 0; oh < a[OH]; oh++) {
                                    int ih = oh * a[STRIDE] - a[PAD] + kh * a[DIL];
                                    if (ih < 0 || ih >= a[H]) {
                                        continue;
                                    }
                                    for (int ow = 0; ow < a[OW]; ow++) {
                                        int iw = ow * a[STRIDE] - a[PAD] + kw * a[DIL];
                                        if (iw < 0 || iw >= a[W]) {
                                            continue;
                                        }
                                        acc += (double) dy.get(((n * a[COUT] + co) * a[OH] + oh) * a[OW] + ow) * x.get(((n * a[CIN] + g * cg + ci) * a[H] + ih) * a[W] + iw);
                                    }
                                }
                            }
                            dw.set(((co * cg + ci) * k + kh) * k + kw, (float) acc);
                        }
                    }
                }
            }
        }
    }
}
