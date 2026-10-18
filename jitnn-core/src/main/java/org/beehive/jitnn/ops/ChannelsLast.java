package org.beehive.jitnn.ops;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.engine.Storage;
import org.beehive.jitnn.kernels.GemmKernels;
import org.beehive.jitnn.kernels.LibraryGlueKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Channels-last inference for convolutions a library runs channels-last (MLX on Metal). Tensors keep
 * their NCHW shape, but those that only convolutions and same-shape element-wise operators read and
 * write are stored [N, H, W, C]: the convolution reads and writes them directly, the element-wise
 * operators do not care, and a layout conversion runs only where an operator needs NCHW (pooling,
 * convolutions on jitNN's kernels, outputs). Without the pass, every library convolution transposes
 * its input and output. {@code -Djitnn.channelsLast=false} turns it off.
 */
public final class ChannelsLast {

    private static final boolean ENABLED = !"false".equals(System.getProperty("jitnn.channelsLast"));
    /** Logs each layout conversion and the operator that needs it ({@code -Djitnn.channelsLast.verbose=true}). */
    private static final boolean VERBOSE = Boolean.getBoolean("jitnn.channelsLast.verbose");

    static final Conv CONV = new Conv();
    static final Winograd WINOGRAD = new Winograd();
    static final Layout TO_HALF = new Layout(2);
    static final Layout TO_LAST = new Layout(1);
    static final Layout TO_FIRST = new Layout(0);

    private static final int N = 0, CIN = 1, H = 2, W = 3, COUT = 4, OH = 5, OW = 6, K = 7, STRIDE = 8, PAD = 9, DIL = 10, GROUPS = 11, BIAS = 12, RELU = 13;

    private ChannelsLast() {
    }

    /** Whether a library takes this convolution channels-last on the current device. */
    private static boolean eligible(Node n) {
        if (n.op != Conv2d.OP) {
            return false;
        }
        int[] a = n.iattr;
        DType t = n.out(0).dtype();
        // an FP16 convolution may read an FP32 input: its conversion to channels-last rounds it
        boolean input = n.in(0).dtype() == t || (t == DType.F16 && n.in(0).dtype() == DType.F32);
        if (t != DType.F32 && t != DType.F16 || !input || n.in(1).dtype() != t) {
            return false;
        }
        return org.beehive.jitnn.natives.Libs.mlxConvChannelsLast(a[N], a[CIN], a[H], a[W], a[COUT], a[K], a[STRIDE], a[OH], a[OW], a[GROUPS], a[DIL], t == DType.F16);
    }

    /**
     * A convolution jitNN runs as Winograd F(4x4, 3x3) that MLX does not take (the 3x3 stride-1 shapes
     * MLX would run with Winograd): its transforms run channels-last too, so no conversion surrounds it.
     */
    private static boolean winograd(Node n) {
        if (n.op != Conv2d.OP) {
            return false;
        }
        int[] a = n.iattr;
        DType t = n.out(0).dtype();
        boolean types = (t == DType.F32 || t == DType.F16) && (n.in(0).dtype() == DType.F32 || n.in(0).dtype() == DType.F16)
                && (n.in(1).dtype() == DType.F32 || n.in(1).dtype() == DType.F16);
        return types && Conv2d.winograd(a, a[OH], a[OW]) && Conv2d.winograd4(a[N], a[OH], a[OW]);
    }

    /** A same-shape element-wise operator or cast over rank-4 tensors: indifferent to the layout. */
    private static boolean elementwise(Node n) {
        if ((n.op != Unary.OP && n.op != Binary.OP && n.op != Cast.OP) || n.outputs.length != 1 || n.out(0).rank() != 4) {
            return false;
        }
        for (Tensor t : n.inputs) {
            // a cast changes the type, not the order of the elements
            if (!t.shape().equals(n.out(0).shape()) || (n.op != Cast.OP && t.dtype() != n.out(0).dtype())) {
                return false;
            }
        }
        return true;
    }

    private static boolean pinned(Tensor t, Set<Storage> wanted) {
        return wanted.contains(t.storage()) || t.storage().isPersistent();
    }

    /** The nodes with channels-last storage where it pays; the nodes unchanged when no convolution qualifies. */
    static List<Node> rewrite(List<Node> nodes, Set<Storage> wanted) {
        if (!ENABLED) {
            return nodes;
        }
        boolean any = false;
        for (Node n : nodes) {
            if (eligible(n)) {
                any = true;
                break;
            }
        }
        if (!any) {
            return nodes;
        }
        // storages holding [N, H, W, C] data, and the converted copies made so far
        Set<Storage> last = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        IdentityHashMap<Storage, Tensor> asLast = new IdentityHashMap<>();
        IdentityHashMap<Storage, Tensor> asFirst = new IdentityHashMap<>();
        IdentityHashMap<Storage, Tensor> asLastHalf = new IdentityHashMap<>();
        IdentityHashMap<Storage, Tensor> castHalf = new IdentityHashMap<>();
        // [N, C, H W] of each channels-last storage (readers may see it through views of other shapes)
        IdentityHashMap<Storage, int[]> dims = new IdentityHashMap<>();
        List<Node> out = new ArrayList<>(nodes.size() + 16);
        for (Node n : nodes) {
            if (eligible(n)) {
                int[] a = n.iattr;
                Tensor x = n.in(0);
                int[] xd = { a[N], a[CIN], a[H] * a[W] };
                Tensor xl;
                if (x.dtype() != n.out(0).dtype() && last.contains(x.storage())) {
                    xl = castToHalf(x, xd, castHalf, out);
                } else if (x.dtype() != n.out(0).dtype()) {
                    xl = convertToHalf(x, xd, asLastHalf, out);
                } else {
                    xl = last.contains(x.storage()) ? x : convert(x, xd, true, asLast, out);
                }
                Tensor y = n.out(0);
                int[] yd = { a[N], a[COUT], a[OH] * a[OW] };
                Tensor[] ins = n.inputs.clone();
                ins[0] = xl;
                if (pinned(y, wanted)) {
                    Tensor yl = Tensor.result(y.shape(), y.dtype());
                    out.add(new Node(CONV, ins, new Tensor[] { yl }, n.iattr, n.fattr, true));
                    out.add(new Node(TO_FIRST, new Tensor[] { yl }, new Tensor[] { y }, yd, null, true));
                } else {
                    out.add(new Node(CONV, ins, n.outputs, n.iattr, n.fattr, true));
                    last.add(y.storage());
                    dims.put(y.storage(), yd);
                }
                continue;
            }
            if (winograd(n)) {
                int[] a = n.iattr;
                Tensor x = n.in(0);
                Tensor[] ins = n.inputs.clone();
                ins[0] = last.contains(x.storage()) ? x : convert(x, new int[] { a[N], a[CIN], a[H] * a[W] }, true, asLast, out);
                Tensor y = n.out(0);
                int[] yd = { a[N], a[COUT], a[OH] * a[OW] };
                if (pinned(y, wanted)) {
                    Tensor yl = Tensor.result(y.shape(), y.dtype());
                    out.add(new Node(WINOGRAD, ins, new Tensor[] { yl }, n.iattr, n.fattr, true));
                    out.add(new Node(TO_FIRST, new Tensor[] { yl }, new Tensor[] { y }, yd, null, true));
                } else {
                    out.add(new Node(WINOGRAD, ins, n.outputs, n.iattr, n.fattr, true));
                    last.add(y.storage());
                    dims.put(y.storage(), yd);
                }
                continue;
            }
            boolean anyLast = false;
            for (Tensor t : n.inputs) {
                anyLast |= last.contains(t.storage());
            }
            if (!anyLast) {
                out.add(n);
                continue;
            }
            if (elementwise(n) && !pinned(n.out(0), wanted)) {
                // the operator runs on [N, H, W, C] data: its other inputs are converted to it
                Tensor[] ins = n.inputs.clone();
                for (int i = 0; i < ins.length; i++) {
                    if (!last.contains(ins[i].storage())) {
                        ins[i] = convert(ins[i], layoutAttr(ins[i]), true, asLast, out);
                    }
                }
                out.add(new Node(n.op, ins, n.outputs, n.iattr, n.fattr, true));
                last.add(n.out(0).storage());
                dims.put(n.out(0).storage(), layoutAttr(n.out(0)));
                continue;
            }
            // an operator that needs NCHW reads converted copies
            Tensor[] ins = n.inputs.clone();
            for (int i = 0; i < ins.length; i++) {
                if (last.contains(ins[i].storage())) {
                    if (VERBOSE && !asFirst.containsKey(ins[i].storage())) {
                        System.err.printf("[jitnn] channels-last: %s reads %s in NCHW%n", n, ins[i].shape());
                    }
                    ins[i] = convert(ins[i], dims.get(ins[i].storage()), false, asFirst, out);
                }
            }
            out.add(new Node(n.op, ins, n.outputs, n.iattr, n.fattr, true));
        }
        return out;
    }

    private static int[] layoutAttr(Tensor t) {
        return new int[] { t.dim(0), t.dim(1), t.dim(2) * t.dim(3) };
    }

    /**
     * t in the other layout, of NCHW extents nchw = [N, C, H W]: one conversion per storage and
     * direction, reused by every later reader (the copy keeps t's shape, which may be a view's).
     */
    private static Tensor convert(Tensor t, int[] nchw, boolean toLast, IdentityHashMap<Storage, Tensor> cache, List<Node> out) {
        Tensor c = cache.get(t.storage());
        if (c == null) {
            if (VERBOSE) {
                System.err.printf("[jitnn] channels-last: %s %s %s%n", toLast ? "to NHWC" : "to NCHW", t.dtype(), java.util.Arrays.toString(nchw));
            }
            c = Tensor.result(t.shape(), t.dtype());
            out.add(new Node(toLast ? TO_LAST : TO_FIRST, new Tensor[] { t }, new Tensor[] { c }, nchw, null, true));
            cache.put(t.storage(), c);
        }
        return c;
    }

    /** An FP32 channels-last tensor as FP16 channels-last (nchw = [N, C, H W]), once per storage. */
    private static Tensor castToHalf(Tensor t, int[] nchw, IdentityHashMap<Storage, Tensor> cache, List<Node> out) {
        Tensor c = cache.get(t.storage());
        if (c == null) {
            c = Tensor.result(t.shape(), DType.F16);
            out.add(new Node(TO_HALF, new Tensor[] { t }, new Tensor[] { c }, nchw, null, true));
            cache.put(t.storage(), c);
        }
        return c;
    }

    /** An FP32 NCHW tensor as FP16 NHWC (nchw = [N, C, H W]), once per storage. */
    private static Tensor convertToHalf(Tensor t, int[] nchw, IdentityHashMap<Storage, Tensor> cache, List<Node> out) {
        Tensor c = cache.get(t.storage());
        if (c == null) {
            c = Tensor.result(t.shape(), DType.F16);
            out.add(new Node(TO_LAST, new Tensor[] { t }, new Tensor[] { c }, nchw, null, true));
            cache.put(t.storage(), c);
        }
        return c;
    }

    /**
     * NCHW to NHWC (mode 1, also from FP32 into FP16) or back (mode 0), or FP32 to FP16 in the same
     * layout (mode 2); iattr [N, C, H W].
     */
    static final class Layout implements Op {

        private final int mode;
        private final boolean toLast;

        Layout(int mode) {
            this.mode = mode;
            this.toLast = mode == 1;
        }

        @Override
        public String name() {
            return mode == 2 ? "to_half" : toLast ? "to_channels_last" : "to_channels_first";
        }

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node n) {
            int c = n.iattr[1];
            int hw = n.iattr[2];
            int total = n.iattr[0] * c * hw;
            if (mode == 2) {
                FloatArray x = e.f32(n.in(0));
                HalfFloatArray y = e.f16(n.out(0));
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), x, y, total));
                return;
            }
            if (n.out(0).dtype() == DType.F16 && n.in(0).dtype() == DType.F32) {
                FloatArray x = e.f32(n.in(0));
                HalfFloatArray y = e.f16(n.out(0));
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLastToHalf, e.context(), x, y, c, hw, total));
                return;
            }
            if (n.out(0).dtype() == DType.F16) {
                HalfFloatArray x = e.f16(n.in(0));
                HalfFloatArray y = e.f16(n.out(0));
                if (toLast) {
                    e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLastHalf, e.context(), x, y, c, hw, total));
                } else {
                    FloatArray none = e.dummyF32();
                    e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::fromChannelsLastHalf, e.context(), x, y, none, c, hw, total, 0));
                }
                return;
            }
            FloatArray x = e.f32(n.in(0));
            FloatArray y = e.f32(n.out(0));
            if (toLast) {
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLast, e.context(), x, y, c, hw, total));
            } else {
                FloatArray none = e.dummyF32();
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::fromChannelsLast, e.context(), x, y, none, c, hw, total, 0));
            }
        }

        @Override
        public void reference(Node n) {
            throw new UnsupportedOperationException(name() + " runs only in compiled plans");
        }
    }

    /** {@link Conv2d} on [N, H, W, C] input and output (iattr as Conv2d's); the weight stays [Cout, Cin, K, K]. */
    static final class Conv implements Op {

        @Override
        public String name() {
            return "conv2d_channels_last";
        }

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            int epilogue = (a[BIAS] != 0 ? GemmKernels.BIAS : 0) | (a[RELU] != 0 ? GemmKernels.RELU : 0);
            FloatArray bias = a[BIAS] != 0 ? e.f32(node.in(2)) : null;
            if (node.out(0).dtype() == DType.F16) {
                org.beehive.jitnn.natives.Libs.conv2dChannelsLastHalf(e, e.f16(node.in(0)), e.f16(node.in(1)), e.f16(node.out(0)), bias, a[N], a[CIN], a[H], a[W], a[COUT], a[OH],
                        a[OW], a[K], a[STRIDE], a[PAD], epilogue);
            } else {
                org.beehive.jitnn.natives.Libs.conv2dChannelsLast(e, e.f32(node.in(0)), e.f32(node.in(1)), e.f32(node.out(0)), bias, a[N], a[CIN], a[H], a[W], a[COUT], a[OH],
                        a[OW], a[K], a[STRIDE], a[PAD], epilogue);
            }
        }

        @Override
        public void reference(Node n) {
            throw new UnsupportedOperationException(name() + " runs only in compiled plans");
        }
    }

    /**
     * {@link Conv2d} as Winograd F(4x4, 3x3) on [N, H, W, C] input and output (iattr as Conv2d's):
     * channels-last input and output transforms around the 36 batched products (MLX where it takes
     * them, jitNN's GEMM otherwise).
     */
    static final class Winograd implements Op {

        @Override
        public String name() {
            return "conv2d_winograd_channels_last";
        }

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public void emit(Emitter e, Node node) {
            int[] a = node.iattr;
            int cin = a[CIN], h = a[H], wd = a[W], cout = a[COUT], oh = a[OH], ow = a[OW], pad = a[PAD];
            int th = Emitter.ceilDiv(oh, 4);
            int tw = Emitter.ceilDiv(ow, 4);
            int tiles = a[N] * th * tw;
            FloatArray v = e.sharedScratch("winogradV", 36 * cin * tiles);
            FloatArray u = e.sharedScratch("winogradU", 36 * cout * cin);
            FloatArray mm = e.sharedScratch("winogradM", 36 * cout * tiles);
            if (node.in(0).dtype() == DType.F16) {
                HalfFloatArray x = e.f16(node.in(0));
                e.task(Emitter.grid1D(cin * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4InputLastHalf, e.context(), x, v, cin, h, wd,
                        pad, th, tw, tiles));
            } else {
                FloatArray x = e.f32(node.in(0));
                e.task(Emitter.grid1D(cin * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4InputLast, e.context(), x, v, cin, h, wd, pad,
                        th, tw, tiles));
            }
            FloatArray w;
            if (node.in(1).dtype() == DType.F16) {
                HalfFloatArray wh = e.f16(node.in(1));
                int nw = cout * cin * 9;
                w = e.sharedScratch("winogradW", nw);
                FloatArray wf = w;
                e.task(Emitter.grid1D(nw), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f16ToF32, e.context(), wh, wf, nw));
            } else {
                w = e.f32(node.in(1));
            }
            FloatArray wt = w;
            e.task(Emitter.grid1D(cout * cin), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4WeightLast, e.context(), wt, u, cout, cin, 0));
            // mm[k] [tiles, cout] = v[k] [tiles, cin] u[k] [cin, cout] for the 36 tile positions k
            int tilesY = Emitter.ceilDiv(tiles, GemmKernels.BM);
            FloatArray none = e.dummyF32();
            if (!org.beehive.jitnn.natives.Libs.gemmBatched(e, v, u, mm, null, tiles, cout, cin, 0, 0, 0, 36, tiles * cin, cin * cout)) {
                e.task(Emitter.blocks2D(Emitter.ceilDiv(cout, GemmKernels.BN), tilesY * 36, 16, 16),
                        (g, id) -> g.task(id, GemmKernels::gemm64pdBatched, e.context(), v, u, mm, none, tiles, cout, cin, 0, 0, 0, tiles * cin, cin * cout, tilesY));
            }
            int epilogue = (a[BIAS] != 0 ? GemmKernels.BIAS : 0) | (a[RELU] != 0 ? GemmKernels.RELU : 0);
            FloatArray bias = a[BIAS] != 0 ? e.f32(node.in(2)) : none;
            if (node.out(0).dtype() == DType.F16) {
                HalfFloatArray y = e.f16(node.out(0));
                e.task(Emitter.grid1D(cout * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4OutputLastHalf, e.context(), mm, y, bias, cout,
                        oh, ow, th, tw, tiles, epilogue));
            } else {
                FloatArray y = e.f32(node.out(0));
                e.task(Emitter.grid1D(cout * tiles), (g, id) -> g.task(id, org.beehive.jitnn.kernels.ConvChannelsLastKernels::winograd4OutputLast, e.context(), mm, y, bias, cout, oh,
                        ow, th, tw, tiles, epilogue));
            }
        }

        @Override
        public void reference(Node n) {
            throw new UnsupportedOperationException(name() + " runs only in compiled plans");
        }
    }
}
