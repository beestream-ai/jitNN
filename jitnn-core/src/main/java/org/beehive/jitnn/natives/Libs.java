package org.beehive.jitnn.natives;

import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Native;
import org.beehive.jitnn.engine.Native.Lib;
import org.beehive.jitnn.kernels.LibraryGlueKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Operators' entry to the vendor libraries. Each method adds the library call (and any epilogue
 * the library does not fuse) and returns true, or adds nothing and returns false when no enabled,
 * available library takes the call; the operator then emits its own kernels.
 */
public final class Libs {

    private Libs() {
    }

    /** A probe of one library function on the current device (null when it works or has no probe of its own). */
    public static String probeFunction(Lib lib, String function) {
        if (lib == Lib.CUDNN && function.equals("sdpa")) {
            return Cuda.probeAttention();
        }
        return lib == Lib.CUDNN && function.equals("convHalf") ? Cuda.probeConvHalf() : null;
    }

    /** Called by {@link Native} to probe a library on the current device: null when it works. */
    public static String probe(Lib lib) {
        return lib == Lib.MLX ? Mlx.probe() : Cuda.probe(lib);
    }

    private static boolean exact(FloatArray a, long numel) {
        return a.getSize() == numel;
    }

    /**
     * Fewest flops for MLX to take operands held in larger arrays (intermediates in reused buffers),
     * through exact-size copies; below it the copies' launches cost more than they save.
     */
    private static final long MLX_COPY_MIN_FLOPS = Long.getLong("jitnn.native.mlxCopyMinFlops", 1L << 30);

    /** Whether MLX can take these operands: exactly sized, or larger and the product big enough to copy them. */
    private static boolean mlxOperands(FloatArray a, FloatArray b, FloatArray c, long na, long nb, long nc, long flops) {
        if (exact(a, na) && exact(b, nb) && exact(c, nc)) {
            return true;
        }
        return flops >= MLX_COPY_MIN_FLOPS && a.getSize() >= na && b.getSize() >= nb && c.getSize() >= nc && nc <= Integer.MAX_VALUE;
    }

    /**
     * FP16 C [m, n] = op(A) op(B) on MLX (op(A) = A; op(B) = B or B^T) into c, or into an exact-size
     * scratch copied to c; false when MLX does not take it. Operands in larger (reused) arrays are
     * copied to exact-size ones when the product is big enough.
     */
    private static boolean mlxHalf(Emitter e, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int m, int n, int k, int ta, int tb) {
        long na = (long) m * k, nb = (long) k * n, nc = (long) m * n;
        boolean exactAll = a.getSize() == na && b.getSize() == nb && c.getSize() == nc;
        boolean copyable = 2L * m * n * k >= MLX_COPY_MIN_FLOPS && a.getSize() >= na && b.getSize() >= nb && c.getSize() >= nc;
        if (ta != 0 || (m == 1 && n == 1) || !(exactAll || copyable) || !Native.use("gemmHalf", Lib.MLX)) {
            return false;
        }
        HalfFloatArray ax = exactCopyHalf(e, a, (int) na);
        HalfFloatArray bx = exactCopyHalf(e, b, (int) nb);
        HalfFloatArray cx = c.getSize() == nc ? c : e.scratchF16((int) nc);
        if (tb == 1) {
            Mlx.matmulTransposedHalf(e, ax, bx, cx, m, n, k);
        } else {
            Mlx.matmulHalf(e, ax, bx, cx, m, n, k);
        }
        if (cx != c) {
            int count = (int) nc;
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LibraryGlueKernels::copyHalf, e.context(), cx, c, count));
        }
        return true;
    }

    private static HalfFloatArray exactCopyHalf(Emitter e, HalfFloatArray a, int n) {
        if (a.getSize() == n) {
            return a;
        }
        HalfFloatArray x = e.scratchF16(n);
        e.task(Emitter.grid1D(n), (g, id) -> g.task(id, LibraryGlueKernels::copyHalf, e.context(), a, x, n));
        return x;
    }

    /** FP32 c = epilogue(FP16 h): an FP16 product into an FP32 result, bias and activation applied. */
    private static void fromHalf(Emitter e, HalfFloatArray h, FloatArray c, FloatArray bias, int m, int n, int epilogue) {
        FloatArray b = bias != null ? bias : e.dummyF32();
        int total = m * n;
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::epilogueFromHalf, e.context(), h, b, c, m, n, total, epilogue));
    }

    /** a itself when it holds exactly n elements, else a copy of its first n into a scratch array. */
    private static FloatArray exactCopy(Emitter e, FloatArray a, int n) {
        if (exact(a, n)) {
            return a;
        }
        FloatArray x = e.scratch(n);
        e.task(Emitter.grid1D(n), (g, id) -> g.task(id, LibraryGlueKernels::copy, e.context(), a, x, n));
        return x;
    }

    /** Applies the GEMM epilogue over {@code batch} [rows, cols] matrices in c (nothing for epilogue 0). */
    private static void epilogue(Emitter e, FloatArray c, FloatArray bias, int rows, int cols, int batch, int epilogue) {
        if (epilogue == 0) {
            return;
        }
        FloatArray b = bias != null ? bias : e.dummyF32();
        int total = rows * cols * batch;
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::epilogueBatched, e.context(), c, b, rows, cols, total, epilogue));
    }

    /** C [m, n] = op(A) op(B) (+ epilogue, see GemmKernels) for FP32 row-major matrices. */
    public static boolean gemm(Emitter e, FloatArray a, FloatArray b, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        if (Native.use("gemm", Lib.CUBLAS)) {
            Cuda.sgemm(e, a, b, c, m, n, k, ta, tb);
        } else if (ta == 0 && !(m == 1 && n == 1) && mlxOperands(a, b, c, (long) m * k, (long) k * n, (long) m * n, 2L * m * n * k) && Native.use("gemm", Lib.MLX)) {
            FloatArray ax = exactCopy(e, a, m * k);
            FloatArray bx = exactCopy(e, b, k * n);
            FloatArray cx = exact(c, (long) m * n) ? c : e.scratch(m * n);
            if (tb == 1) {
                Mlx.matmulTransposed(e, ax, bx, cx, m, n, k);
            } else {
                Mlx.matmul(e, ax, bx, cx, m, n, k);
            }
            if (cx != c) {
                int count = m * n;
                e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LibraryGlueKernels::copy, e.context(), cx, c, count));
            }
        } else {
            return false;
        }
        epilogue(e, c, bias, m, n, 1, epilogue);
        return true;
    }

    /**
     * C[z] [m, n] = op(A[z]) op(B[z]) (+ epilogue) for z < batch: A[z] at z * strideA and B[z] at
     * z * strideB (0: the same matrix for every z), C contiguous.
     */
    public static boolean gemmBatched(Emitter e, FloatArray a, FloatArray b, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue, int batch,
            int strideA, int strideB) {
        if (Native.use("gemm", Lib.CUBLAS)) {
            Cuda.sgemmBatched(e, a, b, c, m, n, k, ta, tb, batch, strideA, strideB);
        } else if (ta == 0 && tb == 0 && strideA == m * k && strideB == k * n && exact(a, (long) batch * m * k) && exact(b, (long) batch * k * n)
                && exact(c, (long) batch * m * n) && Native.use("gemm", Lib.MLX)) {
            Mlx.matmulBatched(e, a, b, c, batch, m, n, k);
        } else {
            return false;
        }
        epilogue(e, c, bias, m, n, batch, epilogue);
        return true;
    }

    /**
     * Exact GELU after an FP16 product runs as cuBLASLt's tanh-approximated GELU epilogue unless
     * {@code -Djitnn.native.approximateGelu=false} (the difference, at most about 5e-4, is below
     * FP16's rounding of the outputs).
     */
    private static final boolean APPROXIMATE_GELU = !"false".equals(System.getProperty("jitnn.native.approximateGelu"));

    /** The bias of FP16 products in cuBLASLt's epilogue unless {@code -Djitnn.native.biasEpilogue=false}. */
    private static final boolean BIAS_EPILOGUE = !"false".equals(System.getProperty("jitnn.native.biasEpilogue"));

    /** FP16 C [m, n] = op(A) op(B) (+ epilogue, applied in FP32) for FP16 A and B. */
    public static boolean gemmHalfOut(Emitter e, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        if (!Native.use("gemm", Lib.CUBLAS)) {
            // Metal: MLX's FP16 product, then jitNN's FP16 epilogue
            if (!mlxHalf(e, a, b, c, m, n, k, ta, tb)) {
                return false;
            }
            if (epilogue != 0) {
                FloatArray bb = bias != null ? bias : e.dummyF32();
                int total = m * n;
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::epilogueHalf, e.context(), c, bb, m, n, total, epilogue));
            }
            return true;
        }
        int bias1 = org.beehive.jitnn.kernels.GemmKernels.BIAS;
        int relu = org.beehive.jitnn.kernels.GemmKernels.RELU;
        int gelu = org.beehive.jitnn.kernels.GemmKernels.GELU;
        int geluTanh = org.beehive.jitnn.kernels.GemmKernels.GELU_TANH;
        if (BIAS_EPILOGUE && bias != null && ta == 0 && tb == 1 && (epilogue & bias1) != 0 && Native.use("gemmBias", Lib.CUBLAS)) {
            // the bias, and ReLU or GELU (tanh-approximated: the exact one too, unless
            // -Djitnn.native.approximateGelu=false), in cuBLASLt's epilogue, on an FP16 copy of the bias
            HalfFloatArray bh = e.scratchF16(n);
            e.task(Emitter.grid1D(n), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), bias, bh, n));
            int act = epilogue & ~bias1;
            boolean fuseGelu = act == geluTanh || (act == gelu && APPROXIMATE_GELU);
            boolean fuseRelu = act == relu;
            Cuda.gemmHalfBias(e, a, b, c, bh, m, n, k, fuseRelu, fuseGelu);
            int after = fuseGelu || fuseRelu ? 0 : act;
            if (after != 0) {
                int total = m * n;
                FloatArray none = e.dummyF32();
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::epilogueHalf, e.context(), c, none, m, n, total, after));
            }
            return true;
        }
        Cuda.gemmHalf(e, a, b, c, m, n, k, ta, tb);
        if (epilogue != 0) {
            FloatArray bb = bias != null ? bias : e.dummyF32();
            int total = m * n;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::epilogueHalf, e.context(), c, bb, m, n, total, epilogue));
        }
        return true;
    }

    /**
     * FP16 C [m, n] = A W^T + bias (+ ReLU or GELU) with an FP16 bias, all in cuBLASLt's epilogue;
     * false when cuBLASLt's epilogue does not take it (the caller converts the bias and retries).
     */
    public static boolean gemmHalfOutHalfBias(Emitter e, HalfFloatArray a, HalfFloatArray w, HalfFloatArray c, HalfFloatArray bias, int m, int n, int k, int epilogue) {
        int bias1 = org.beehive.jitnn.kernels.GemmKernels.BIAS;
        int act = epilogue & ~bias1;
        boolean fuseGelu = act == org.beehive.jitnn.kernels.GemmKernels.GELU_TANH || (act == org.beehive.jitnn.kernels.GemmKernels.GELU && APPROXIMATE_GELU);
        boolean fuseRelu = act == org.beehive.jitnn.kernels.GemmKernels.RELU;
        if (!BIAS_EPILOGUE || (epilogue & bias1) == 0 || !(act == 0 || fuseGelu || fuseRelu) || !Native.use("gemm", Lib.CUBLAS) || !Native.use("gemmBias", Lib.CUBLAS)) {
            return false;
        }
        Cuda.gemmHalfBias(e, a, w, c, bias, m, n, k, fuseRelu, fuseGelu);
        return true;
    }

    /** FP32 C [m, n] = op(A) op(B) (+ epilogue) for FP16 A and B. */
    public static boolean gemmF16(Emitter e, HalfFloatArray a, HalfFloatArray b, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        if (!Native.use("gemm", Lib.CUBLAS)) {
            // Metal: MLX's FP16 product (an FP16 result, as PyTorch's autocast gives), widened with the epilogue
            HalfFloatArray h = e.scratchF16(m * n);
            if (!mlxHalf(e, a, b, h, m, n, k, ta, tb)) {
                return false;
            }
            fromHalf(e, h, c, bias, m, n, epilogue);
            return true;
        }
        Cuda.gemmF16(e, a, b, c, m, n, k, ta, tb);
        epilogue(e, c, bias, m, n, 1, epilogue);
        return true;
    }

    /**
     * FP32 C [m, n] = A W^T (+ epilogue) for an FP32 A [m, k] and an FP16 weight W [n, k]: A is
     * rounded to FP16 (as jitNN's tensor-core kernel does when it stages it), then one cuBLAS call.
     */
    public static boolean gemmA32(Emitter e, FloatArray a, HalfFloatArray w, FloatArray c, FloatArray bias, int m, int n, int k, int epilogue) {
        if (!Native.use("gemm", Lib.CUBLAS)) {
            // Metal: A rounded to FP16, MLX's FP16 product, widened with the epilogue
            if (a.getSize() < (long) m * k || w.getSize() < (long) n * k || (m == 1 && n == 1) || !Native.use("gemmHalf", Lib.MLX)) {
                return false;
            }
            HalfFloatArray ah = e.scratchF16(m * k);
            int count = m * k;
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), a, ah, count));
            HalfFloatArray h = e.scratchF16(m * n);
            // exact-size A and C: the product only declines a larger weight below the copy threshold
            if (!mlxHalf(e, ah, w, h, m, n, k, 0, 1)) {
                HalfFloatArray wx = e.scratchF16(n * k);
                int wn = n * k;
                e.task(Emitter.grid1D(wn), (g, id) -> g.task(id, LibraryGlueKernels::copyHalf, e.context(), w, wx, wn));
                Mlx.matmulTransposedHalf(e, ah, wx, h, m, n, k);
            }
            fromHalf(e, h, c, bias, m, n, epilogue);
            return true;
        }
        HalfFloatArray ah = e.scratchF16(m * k);
        int count = m * k;
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), a, ah, count));
        Cuda.gemmF16(e, ah, w, c, m, n, k, 0, 1);
        epilogue(e, c, bias, m, n, 1, epilogue);
        return true;
    }

    /**
     * Fewest multiply-adds (2 x N x Cout x OH x OW x Cin x K x K) for a convolution to go to cuDNN
     * ({@code -Djitnn.native.convMinFlops}): smaller ones keep jitNN's kernels.
     */
    private static final long CONV_MIN_FLOPS = Long.getLong("jitnn.native.convMinFlops", 1L << 30);

    /**
     * out [n, cout, oh, ow] = conv(x [n, cin, h, w], w [cout, cin, r, r]) (+ per-channel bias, ReLU
     * per epilogue BIAS / RELU) for one group and no dilation.
     */
    public static boolean conv2d(Emitter e, FloatArray x, FloatArray w, FloatArray out, FloatArray bias, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride,
            int pad, int dil, int groups, int epilogue) {
        long flops = 2L * n * cout * oh * ow * cin * r * r;
        if (groups != 1 || dil != 1) {
            return false;
        }
        if (flops >= MLX_CONV_MIN_FLOPS && mlxConvShape(n, cin, h, wd, cout, r, stride) && Native.use("conv", Lib.MLX)) {
            // MLX convolves channels-last: x and the weight transposed in, the result transposed back with the epilogue
            FloatArray xl = e.sharedScratch("mlxConvIn", n * cin * h * wd);
            FloatArray wl = e.sharedScratch("mlxConvWeight", cout * cin * r * r);
            FloatArray yl = e.sharedScratch("mlxConvOut", n * cout * oh * ow);
            int nx = n * cin * h * wd;
            int nw = cout * cin * r * r;
            int ny = n * cout * oh * ow;
            int hw = h * wd;
            int kk = r * r;
            int ohw = oh * ow;
            FloatArray b = bias != null ? bias : e.dummyF32();
            e.task(Emitter.grid1D(nx), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLast, e.context(), x, xl, cin, hw, nx));
            e.task(Emitter.grid1D(nw), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLast, e.context(), w, wl, cin, kk, nw));
            Mlx.conv2d(e, xl, wl, yl, n, h, wd, cin, cout, r, stride, pad);
            e.task(Emitter.grid1D(ny), (g, id) -> g.task(id, LibraryGlueKernels::fromChannelsLast, e.context(), yl, out, b, cout, ohw, ny, epilogue));
            return true;
        }
        if (flops < CONV_MIN_FLOPS || !Native.use("conv", Lib.CUDNN)) {
            return false;
        }
        Cuda.conv2d(e, x, w, out, n, cin, h, wd, cout, r, pad, stride);
        // the bias is per output channel: a row bias over [cout, oh * ow] per image
        int rowEpilogue = ((epilogue & org.beehive.jitnn.kernels.GemmKernels.BIAS) != 0 ? org.beehive.jitnn.kernels.GemmKernels.ROW_BIAS : 0)
                | (epilogue & ~org.beehive.jitnn.kernels.GemmKernels.BIAS);
        epilogue(e, out, bias, cout, oh * ow, n, rowEpilogue);
        return true;
    }

    /** Fewest multiply-adds for a convolution to go to MLX ({@code -Djitnn.native.mlxConvMinFlops}): below it the layout transposes cost more than they save. */
    private static final long MLX_CONV_MIN_FLOPS = Long.getLong("jitnn.native.mlxConvMinFlops", 1L << 28);

    /**
     * Whether TornadoVM's MLX convolution reproduces MLX's kernel choice for this shape (one group, no
     * dilation): the implicit-GEMM kernels for channel counts they specialise (inputs of at most 4 or
     * a multiple of 16 channels, outputs of at most 16 or a multiple of 16), but not the 3x3
     * stride-1 shapes MLX runs with Winograd, which the binding does not reproduce.
     */
    static boolean mlxConvShape(int n, int c, int h, int w, int o, int k, int stride) {
        boolean winograd = stride == 1 && k == 3 && c % 32 == 0 && o % 32 == 0 && (long) n * h * w >= 4096 && c + o >= 256;
        boolean specialized = (c <= 4 || c % 16 == 0) && (o <= 16 || o % 16 == 0);
        return specialized && !winograd;
    }

    /**
     * Whether a convolution of this shape runs as an FP16 library convolution in half-precision
     * inference: one group, no dilation, and cuDNN's FP16 convolution, or MLX's for the shapes and
     * sizes it takes (see {@link #mlxConvShape}). Decided when the convolution is recorded, as it sets
     * the output type.
     */
    public static boolean conv2dHalfAvailable(int n, int c, int h, int w, int o, int k, int stride, int oh, int ow, int groups, int dil) {
        if (groups != 1 || dil != 1) {
            return false;
        }
        if (Native.use("convHalf", Lib.CUDNN)) {
            return true;
        }
        long flops = 2L * n * o * oh * ow * c * k * k;
        return flops >= MLX_CONV_MIN_FLOPS && mlxConvShape(n, c, h, w, o, k, stride) && Native.use("convHalf", Lib.MLX);
    }

    /**
     * FP16 out [n, cout, oh, ow] = conv(x [n, cin, h, w], w [cout, cin, r, r]) (+ per-channel FP32
     * bias, ReLU) with FP32 accumulation; false when cuDNN's FP16 convolution is off.
     */
    public static boolean conv2dHalf(Emitter e, HalfFloatArray x, HalfFloatArray w, HalfFloatArray out, FloatArray bias, int n, int cin, int h, int wd, int cout, int oh, int ow,
            int r, int stride, int pad, int epilogue) {
        if (!Native.use("convHalf", Lib.CUDNN)) {
            long flops = 2L * n * cout * oh * ow * cin * r * r;
            if (flops < MLX_CONV_MIN_FLOPS || !mlxConvShape(n, cin, h, wd, cout, r, stride) || !Native.use("convHalf", Lib.MLX)) {
                return false;
            }
            // MLX, channels-last: x and the weight transposed in, the result transposed back with the epilogue
            int nx = n * cin * h * wd;
            int nw = cout * cin * r * r;
            int ny = n * cout * oh * ow;
            int hw = h * wd;
            int kk = r * r;
            int ohw = oh * ow;
            HalfFloatArray xl = e.sharedScratchF16("mlxConvIn", nx);
            HalfFloatArray wl = e.sharedScratchF16("mlxConvWeight", nw);
            HalfFloatArray yl = e.sharedScratchF16("mlxConvOut", ny);
            FloatArray b = bias != null ? bias : e.dummyF32();
            e.task(Emitter.grid1D(nx), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLastHalf, e.context(), x, xl, cin, hw, nx));
            e.task(Emitter.grid1D(nw), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLastHalf, e.context(), w, wl, cin, kk, nw));
            Mlx.conv2dHalf(e, xl, wl, yl, n, h, wd, cin, cout, r, stride, pad);
            e.task(Emitter.grid1D(ny), (g, id) -> g.task(id, LibraryGlueKernels::fromChannelsLastHalf, e.context(), yl, out, b, cout, ohw, ny, epilogue));
            return true;
        }
        Cuda.conv2dHalf(e, x, w, out, n, cin, h, wd, cout, r, pad, stride);
        if (epilogue != 0) {
            int rowEpilogue = ((epilogue & org.beehive.jitnn.kernels.GemmKernels.BIAS) != 0 ? org.beehive.jitnn.kernels.GemmKernels.ROW_BIAS : 0)
                    | (epilogue & ~org.beehive.jitnn.kernels.GemmKernels.BIAS);
            FloatArray b = bias != null ? bias : e.dummyF32();
            int plane = oh * ow;
            int total = n * cout * plane;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::epilogueHalf, e.context(), out, b, cout, plane, total, rowEpilogue));
        }
        return true;
    }

    /**
     * Whether MLX takes this convolution channels-last on the current device (one group, no dilation,
     * a shape the binding reproduces, at least {@code jitnn.native.mlxConvMinFlops} multiply-adds),
     * in FP16 when {@code half}.
     */
    public static boolean mlxConvChannelsLast(int n, int c, int h, int w, int o, int k, int stride, int oh, int ow, int groups, int dil, boolean half) {
        if (groups != 1 || dil != 1) {
            return false;
        }
        long flops = 2L * n * o * oh * ow * c * k * k;
        return flops >= MLX_CONV_MIN_FLOPS && mlxConvShape(n, c, h, w, o, k, stride) && Native.use(half ? "convHalf" : "conv", Lib.MLX);
    }

    /**
     * out [n, oh, ow, cout] = conv(x [n, h, w, cin], w [cout, cin, r, r]) (+ per-channel bias, ReLU) on
     * MLX, input and output channels-last; the weight is transposed to [cout, r, r, cin] for MLX.
     */
    public static void conv2dChannelsLast(Emitter e, FloatArray x, FloatArray w, FloatArray out, FloatArray bias, int n, int cin, int h, int wd, int cout, int oh, int ow, int r,
            int stride, int pad, int epilogue, boolean weightLast) {
        int nx = n * cin * h * wd;
        int nw = cout * cin * r * r;
        int ny = n * cout * oh * ow;
        int kk = r * r;
        FloatArray xl = exact(x, nx) ? x : e.sharedScratch("mlxConvIn", nx);
        if (xl != x) {
            e.task(Emitter.grid1D(nx), (g, id) -> g.task(id, LibraryGlueKernels::copy, e.context(), x, xl, nx));
        }
        // a weight already [cout, r, r, cin] (cached channels-last copy) goes straight to MLX
        FloatArray wl = weightLast ? w : e.sharedScratch("mlxConvWeight", nw);
        if (!weightLast) {
            e.task(Emitter.grid1D(nw), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLast, e.context(), w, wl, cin, kk, nw));
        }
        FloatArray yl = exact(out, ny) ? out : e.sharedScratch("mlxConvOut", ny);
        Mlx.conv2d(e, xl, wl, yl, n, h, wd, cin, cout, r, stride, pad);
        if (yl != out) {
            e.task(Emitter.grid1D(ny), (g, id) -> g.task(id, LibraryGlueKernels::copy, e.context(), yl, out, ny));
        }
        // channels-last: the bias is per column of [n oh ow, cout]
        epilogue(e, out, bias, n * oh * ow, cout, 1, epilogue);
    }

    /** {@link #conv2dChannelsLast} on FP16 arrays (FP32 bias). */
    public static void conv2dChannelsLastHalf(Emitter e, HalfFloatArray x, HalfFloatArray w, HalfFloatArray out, FloatArray bias, int n, int cin, int h, int wd, int cout, int oh,
            int ow, int r, int stride, int pad, int epilogue, boolean weightLast) {
        int nx = n * cin * h * wd;
        int nw = cout * cin * r * r;
        int ny = n * cout * oh * ow;
        int kk = r * r;
        HalfFloatArray xl = exactCopyHalf(e, x, nx);
        HalfFloatArray wl = weightLast ? w : e.sharedScratchF16("mlxConvWeight", nw);
        if (!weightLast) {
            e.task(Emitter.grid1D(nw), (g, id) -> g.task(id, LibraryGlueKernels::toChannelsLastHalf, e.context(), w, wl, cin, kk, nw));
        }
        HalfFloatArray yl = out.getSize() == ny ? out : e.sharedScratchF16("mlxConvOut", ny);
        Mlx.conv2dHalf(e, xl, wl, yl, n, h, wd, cin, cout, r, stride, pad);
        if (yl != out) {
            e.task(Emitter.grid1D(ny), (g, id) -> g.task(id, LibraryGlueKernels::copyHalf, e.context(), yl, out, ny));
        }
        if (epilogue != 0) {
            FloatArray b = bias != null ? bias : e.dummyF32();
            int rows = n * oh * ow;
            e.task(Emitter.grid1D(ny), (g, id) -> g.task(id, LibraryGlueKernels::epilogueHalf, e.context(), out, b, rows, cout, ny, epilogue));
        }
    }

    /** y = softmax(x) over the rows of x [rows, cols] (not the log form). */
    public static boolean softmax(Emitter e, FloatArray x, FloatArray y, int rows, int cols) {
        if (Native.use("softmax", Lib.CUDNN)) {
            Cuda.softmax(e, x, y, rows, cols);
            return true;
        }
        if (exact(x, (long) rows * cols) && exact(y, (long) rows * cols) && Native.use("softmax", Lib.MLX)) {
            Mlx.softmax(e, x, y, rows, cols);
            return true;
        }
        return false;
    }

    /**
     * A full-length FFT over {@code batch} contiguous rows: complex to complex (forward for sign -1,
     * inverse for +1), or real to complex (forward, n / 2 + 1 outputs), then times {@code scale}.
     */
    public static boolean fft(Emitter e, FloatArray x, FloatArray y, int n, int batch, int sign, boolean realIn, float scale) {
        if (!Native.use("fft", Lib.CUFFT)) {
            return false;
        }
        if (realIn) {
            Cuda.fftR2C(e, x, y, n, batch);
        } else {
            Cuda.fftC2C(e, x, y, n, batch, sign > 0);
        }
        if (scale != 1.0f) {
            int total = batch * 2 * (realIn ? n / 2 + 1 : n);
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LibraryGlueKernels::scale, e.context(), y, scale, total));
        }
        return true;
    }

    /** y = LayerNorm (or RMSNorm without bias) of x [rows, cols], when its saved statistics are not needed. */
    public static boolean rowNorm(Emitter e, FloatArray x, FloatArray w, FloatArray b, FloatArray y, int rows, int cols, float eps, boolean layer) {
        long size = (long) rows * cols;
        if (!exact(x, size) || !exact(y, size) || !exact(w, cols) || (layer && (b == null || !exact(b, cols))) || !Native.use("norm", Lib.MLX)) {
            return false;
        }
        if (layer) {
            Mlx.layerNorm(e, x, w, b, y, rows, cols, eps);
        } else {
            Mlx.rmsNorm(e, x, w, y, rows, cols, eps);
        }
        return true;
    }

    /** cuDNN's FP16 attention for FP32 tensors ({@code -Djitnn.native.attentionFp16=true}): rounds q, k and v to FP16. */
    private static final boolean ATTENTION_FP16 = Boolean.getBoolean("jitnn.native.attentionFp16");

    /**
     * o = softmax(scale q k^T (+ causal mask)) v for contiguous q, o [batch, heads, t, d] and k, v
     * [batch, heads, s, d], without a mask tensor and when the log-sum-exp is not needed (inference):
     * MLX on Metal (FP32), or cuDNN's fused FP16 attention on CUDA when enabled.
     */
    public static boolean attention(Emitter e, FloatArray q, FloatArray k, FloatArray v, FloatArray o, int batch, int heads, int t, int s, int d, float scale, boolean causal) {
        if (causal && t != s) {
            return false;
        }
        long qn = (long) batch * heads * t * d;
        long kn = (long) batch * heads * s * d;
        boolean decode = t <= 8 && (d == 64 || d == 96 || d == 128 || d == 256);
        boolean prefill = t > 8 && (d == 64 || d == 72 || d == 80 || d == 96 || d == 128);
        if ((decode || prefill) && exact(q, qn) && exact(o, qn) && exact(k, kn) && exact(v, kn) && Native.use("sdpa", Lib.MLX)) {
            Mlx.attention(e, q, k, v, o, batch, heads, t, s, d, scale, causal);
            return true;
        }
        if (ATTENTION_FP16 && d % 8 == 0 && d <= 256 && Native.use("sdpa", Lib.CUDNN)) {
            int qc = (int) qn;
            int kc = (int) kn;
            HalfFloatArray qh = e.scratchF16(qc);
            HalfFloatArray kh = e.scratchF16(kc);
            HalfFloatArray vh = e.scratchF16(kc);
            HalfFloatArray oh = e.scratchF16(qc);
            e.task(Emitter.grid1D(qc), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), q, qh, qc));
            e.task(Emitter.grid1D(kc), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), k, kh, kc));
            e.task(Emitter.grid1D(kc), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), v, vh, kc));
            Cuda.attentionF16(e, qh, kh, vh, oh, batch, heads, t, s, d, scale, causal);
            e.task(Emitter.grid1D(qc), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f16ToF32, e.context(), oh, o, qc));
            return true;
        }
        return false;
    }

    /** Fewest q k^T multiply-adds (z t s d) for attention in another layout to be gathered for MLX. */
    private static final long MLX_GATHER_MIN = Long.getLong("jitnn.native.mlxGatherMin", 1L << 26);

    /**
     * Whether attention of this shape held in another layout (packed or [B, T, H, d]) is worth gathering
     * into [B, H, T, d] for MLX's fused attention; {@link #attention} then takes the gathered arrays.
     */
    public static boolean mlxAttentionAfterGather(int z, int t, int s, int d, boolean causal) {
        if (causal && t != s) {
            return false;
        }
        boolean decode = t <= 8 && (d == 64 || d == 96 || d == 128 || d == 256);
        boolean prefill = t > 8 && (d == 64 || d == 72 || d == 80 || d == 96 || d == 128);
        return (decode || prefill) && (long) z * t * s * d >= MLX_GATHER_MIN && Native.use("sdpa", Lib.MLX);
    }

    /** {@link #mlxAttentionAfterGather} for FP16 tensors (MLX's FP16 fused attention). */
    public static boolean mlxHalfAttentionAfterGather(int z, int t, int s, int d, boolean causal) {
        if (causal && t != s) {
            return false;
        }
        boolean decode = t <= 8 && (d == 64 || d == 96 || d == 128 || d == 256);
        boolean prefill = t > 8 && (d == 64 || d == 72 || d == 80 || d == 96 || d == 128);
        return (decode || prefill) && (long) z * t * s * d >= MLX_GATHER_MIN && Native.use("sdpaHalf", Lib.MLX);
    }

    /** MLX's FP16 fused attention on contiguous q [b, h, t, d], k and v [b, h, s, d]; false when it declines. */
    public static boolean mlxAttentionHalf(Emitter e, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, HalfFloatArray o, int batch, int heads, int t, int s, int d,
            float scale, boolean causal) {
        if (!Native.use("sdpaHalf", Lib.MLX)) {
            return false;
        }
        Mlx.attentionHalf(e, q, k, v, o, batch, heads, t, s, d, scale, causal);
        return true;
    }

    /** True when cuDNN's fused FP16 attention may be used (cuDNN available and its attention not turned off). */
    public static boolean fusedAttentionAvailable() {
        return Native.use("sdpa", Lib.CUDNN);
    }

    /** cuDNN's fused FP16 attention on contiguous q, o [b, h, t, d] and k, v [b, h, s, d] (inference). */
    public static void fusedAttention(Emitter e, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, HalfFloatArray o, int b, int h, int t, int s, int d, float scale, boolean causal) {
        Cuda.attentionF16(e, q, k, v, o, b, h, t, s, d, scale, causal);
    }

    /** True when CUTLASS's batched FP16 GEMM is available (FP16 attention on the tensor cores). */
    public static boolean hgemmBatchedAvailable() {
        return Native.use("gemm", Lib.CUTLASS);
    }

    /** FP16 C[z] = A[z] B[z], contiguous [batch, m, k] x [batch, k, n] (n and k multiples of 4): CUTLASS. */
    public static void hgemmBatched(Emitter e, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int m, int n, int k, int batch) {
        Cuda.hgemmBatched(e, a, b, c, m, n, k, batch);
    }

    /** out [rows, n] = A dense for a CSR matrix A [rows, k] of nnz values and a dense row-major [k, n]. */
    public static boolean spmm(Emitter e, IntArray ptr, IntArray col, FloatArray val, FloatArray dense, FloatArray out, int rows, int k, int n, int nnz) {
        if (!Native.use("spmm", Lib.CUSPARSE)) {
            return false;
        }
        Cuda.spmm(e, ptr, col, val, dense, out, rows, k, n, nnz);
        return true;
    }
}
