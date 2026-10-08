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
    /** Winograd F(2x2, 3x3) for 3x3 stride-1 convolutions unless {@code -Djitnn.winograd=false}. */
    static final boolean WINOGRAD = !"false".equals(System.getProperty("jitnn.winograd"));

    /** 1x1 stride-1 convolutions as one GEMM per image unless {@code -Djitnn.conv1x1=implicit}. */
    static final boolean POINTWISE_GEMM = !"implicit".equals(System.getProperty("jitnn.conv1x1"));

    /** Fewest output positions per image for a pointwise convolution to run as one GEMM per image. */
    static final int POINTWISE_MIN_PLANE = 256;

    /**
     * Whether a convolution is a pointwise one that runs as a batched GEMM: 1x1, stride 1, no
     * padding, one group, with at least POINTWISE_MIN_PLANE positions per image (smaller planes
     * leave the 64-wide GEMM tiles partly empty, where the implicit GEMM packs the whole batch) and
     * at least 128 64x64 output tiles over the batch (fewer leave the GPU idle, and the implicit
     * GEMM's split-K handles them).
     */
    static boolean pointwise(int[] a) {
        int plane = a[OH] * a[OW];
        return POINTWISE_GEMM && a[K] == 1 && a[STRIDE] == 1 && a[PAD] == 0 && a[GROUPS] == 1 && plane >= POINTWISE_MIN_PLANE
                && a[N] * Emitter.ceilDiv(a[COUT], GemmKernels.BM) * Emitter.ceilDiv(plane, GemmKernels.BN) >= 128;
    }

    /**
     * Winograd F(2x2, 3x3) filter gradient: dw = G^T (sum over tiles of (A dy A^T) (B^T x B)) G. The
     * tiles are cut into slices that one batched GEMM handles as separate products (more work-groups
     * when the channel counts are small); the last transform sums the slices in order.
     */
    static void emitWinogradWeight(Emitter e, FloatArray dy, FloatArray x, FloatArray dw, int n, int cin, int h, int wd, int cout, int oh, int ow, int pad) {
        int th = Emitter.ceilDiv(oh, 2);
        int tw = Emitter.ceilDiv(ow, 2);
        int tiles = n * th * tw;
        int blocks = 16 * Emitter.ceilDiv(cout, GemmKernels.BM) * Emitter.ceilDiv(cin, GemmKernels.BN);
        int slices = Math.max(1, Math.min(Emitter.ceilDiv(512, blocks), tiles / 256));
        int chunk = Emitter.ceilDiv(Emitter.ceilDiv(tiles, slices), 16) * 16;
        int span = slices * chunk;
        FloatArray v = e.scratch(16 * cin * span);
        FloatArray d = e.scratch(16 * cout * span);
        FloatArray du = e.scratch(16 * slices * cout * cin);
        e.task(Emitter.grid1D(cin * span), (g, id) -> g.task(id, ConvKernels::winogradInput, e.context(), x, v, cin, h, wd, pad, th, tw, tiles, slices, chunk));
        e.task(Emitter.grid1D(cout * span), (g, id) -> g.task(id, ConvKernels::winogradGradOut, e.context(), dy, d, cout, oh, ow, th, tw, tiles, slices, chunk));
        // du[z] = d[z] [cout, chunk] v[z]^T [chunk, cin] for z = (e, slice)
        int tilesY = Emitter.ceilDiv(cout, GemmKernels.BM);
        FloatArray none = e.dummyF32();
        e.task(Emitter.blocks2D(Emitter.ceilDiv(cin, GemmKernels.BN), tilesY * 16 * slices, 16, 16),
                (g, id) -> g.task(id, GemmKernels::gemm64pdBatched, e.context(), d, v, du, none, cout, cin, chunk, 0, 1, 0, cout * chunk, cin * chunk, tilesY));
        e.task(Emitter.grid1D(cout * cin), (g, id) -> g.task(id, ConvKernels::winogradGradWeight, e.context(), du, dw, cout, cin, slices));
    }

    /** Fewest 2x2 output tiles for the Winograd path: below it the batched GEMM leaves the GPU idle. */
    static final int WINOGRAD_MIN_TILES = 128;

    /**
     * Whether a convolution of these attributes takes the Winograd path, producing an oh x ow
     * output per image (the input size for the gradient of the input).
     */
    static boolean winograd(int[] a, int oh, int ow) {
        return WINOGRAD && a[K] == 3 && a[STRIDE] == 1 && a[DIL] == 1 && a[GROUPS] == 1 && a[CIN] >= 16 && a[COUT] >= 16 && a[PAD] <= 2
                && a[N] * Emitter.ceilDiv(oh, 2) * Emitter.ceilDiv(ow, 2) >= WINOGRAD_MIN_TILES;
    }

    /**
     * Winograd F(2x2, 3x3): transforms the input x [n, cin, h, wd] and the filter, multiplies the 16
     * transformed tiles with one batched GEMM, and transforms back into out [n, cout, oh, ow].
     * flip = 1 takes w as [cin, cout, 3, 3] rotated (the gradient of the input).
     */
    static void emitWinograd(Emitter e, FloatArray x, FloatArray w, FloatArray out, FloatArray bias, int n, int cin, int h, int wd, int cout, int oh, int ow, int pad, int flip,
            int epilogue) {
        int th = Emitter.ceilDiv(oh, 2);
        int tw = Emitter.ceilDiv(ow, 2);
        int tiles = n * th * tw;
        FloatArray v = e.scratch(16 * cin * tiles);
        FloatArray u = e.scratch(16 * cout * cin);
        FloatArray mm = e.scratch(16 * cout * tiles);
        e.task(Emitter.grid1D(cin * tiles), (g, id) -> g.task(id, ConvKernels::winogradInput, e.context(), x, v, cin, h, wd, pad, th, tw, tiles, 1, tiles));
        e.task(Emitter.grid1D(cout * cin), (g, id) -> g.task(id, ConvKernels::winogradWeight, e.context(), w, u, cout, cin, flip));
        int tilesY = Emitter.ceilDiv(cout, GemmKernels.BM);
        FloatArray none = e.dummyF32();
        e.task(Emitter.blocks2D(Emitter.ceilDiv(tiles, GemmKernels.BN), tilesY * 16, 16, 16),
                (g, id) -> g.task(id, GemmKernels::gemm64pdBatched, e.context(), u, v, mm, none, cout, tiles, cin, 0, 0, 0, cout * cin, cin * tiles, tilesY));
        e.task(Emitter.grid1D(cout * tiles), (g, id) -> g.task(id, ConvKernels::winogradOutput, e.context(), mm, out, bias, cout, oh, ow, th, tw, tiles, epilogue));
    }

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
        if (pointwise(a)) {
            // out[n] = W [cout, cin] x[n] [cin, OH OW], written straight into NCHW
            int plane = oh * ow;
            int tilesY = Emitter.ceilDiv(cout, GemmKernels.BM);
            int ep = (a[BIAS] != 0 ? GemmKernels.ROW_BIAS : 0) | (a[RELU] != 0 ? GemmKernels.RELU : 0);
            e.task(Emitter.blocks2D(Emitter.ceilDiv(plane, GemmKernels.BN), tilesY * a[N], 16, 16),
                    (g, id) -> g.task(id, GemmKernels::gemm64pdBatched, e.context(), w, x, out, bias, cout, plane, cin, 0, 0, ep, 0, cin * h * wd, tilesY));
            return;
        }
        if (winograd(a, oh, ow)) {
            emitWinograd(e, x, w, out, bias, a[N], cin, h, wd, cout, oh, ow, pd, 0, epilogue);
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
            if (winograd(a, h, wd)) {
                // the input gradient is the 3x3 convolution of dy with the flipped, transposed filter
                emitWinograd(e, dy, w, dx, e.dummyF32(), a[N], cout, oh, ow, cin, h, wd, 2 - pd, 1, 0);
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
            if (winograd(a, oh, ow)) {
                emitWinogradWeight(e, dy, x, dw, a[N], cin, h, wd, cout, oh, ow, pd);
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
