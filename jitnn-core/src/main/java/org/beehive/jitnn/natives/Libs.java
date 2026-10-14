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

    /** Called by {@link Native} to probe a library on the current device: null when it works. */
    public static String probe(Lib lib) {
        return lib == Lib.MLX ? Mlx.probe() : Cuda.probe(lib);
    }

    private static boolean exact(FloatArray a, long numel) {
        return a.getSize() == numel;
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
        } else if (ta == 0 && !(m == 1 && n == 1) && exact(a, (long) m * k) && exact(b, (long) k * n) && exact(c, (long) m * n) && Native.use("gemm", Lib.MLX)) {
            if (tb == 1) {
                Mlx.matmulTransposed(e, a, b, c, m, n, k);
            } else {
                Mlx.matmul(e, a, b, c, m, n, k);
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

    /** The bias of FP16 products in cuBLASLt's epilogue unless {@code -Djitnn.native.biasEpilogue=false}. */
    private static final boolean BIAS_EPILOGUE = !"false".equals(System.getProperty("jitnn.native.biasEpilogue"));

    /** FP16 C [m, n] = op(A) op(B) (+ epilogue, applied in FP32) for FP16 A and B. */
    public static boolean gemmHalfOut(Emitter e, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        if (!Native.use("gemm", Lib.CUBLAS)) {
            return false;
        }
        int bias1 = org.beehive.jitnn.kernels.GemmKernels.BIAS;
        int relu = org.beehive.jitnn.kernels.GemmKernels.RELU;
        if (BIAS_EPILOGUE && bias != null && ta == 0 && tb == 1 && (epilogue & bias1) != 0 && Native.use("gemmBias", Lib.CUBLAS)) {
            // the bias (and ReLU) in cuBLASLt's epilogue, on an FP16 copy of the bias; GELU stays a separate pass
            HalfFloatArray bh = e.scratchF16(n);
            e.task(Emitter.grid1D(n), (g, id) -> g.task(id, org.beehive.jitnn.kernels.LowPrecision::f32ToF16, e.context(), bias, bh, n));
            int rest = epilogue & ~bias1 & ~relu;
            Cuda.gemmHalfBias(e, a, b, c, bh, m, n, k, (epilogue & relu) != 0 && rest == 0);
            int after = (epilogue & relu) != 0 && rest != 0 ? rest | relu : rest;
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

    /** FP32 C [m, n] = op(A) op(B) (+ epilogue) for FP16 A and B. */
    public static boolean gemmF16(Emitter e, HalfFloatArray a, HalfFloatArray b, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        if (!Native.use("gemm", Lib.CUBLAS)) {
            return false;
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
            return false;
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
     * ({@code -Djitnn.native.convMinFlops}): smaller ones keep jitNN's kernels, so that plans of small
     * batches keep their CUDA graph (plans with cuDNN calls run without one, see TornadoExecutor).
     */
    private static final long CONV_MIN_FLOPS = Long.getLong("jitnn.native.convMinFlops", 1L << 30);

    /**
     * out [n, cout, oh, ow] = conv(x [n, cin, h, w], w [cout, cin, r, r]) (+ per-channel bias, ReLU
     * per epilogue BIAS / RELU) for one group and no dilation.
     */
    public static boolean conv2d(Emitter e, FloatArray x, FloatArray w, FloatArray out, FloatArray bias, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride,
            int pad, int dil, int groups, int epilogue) {
        long flops = 2L * n * cout * oh * ow * cin * r * r;
        if (groups != 1 || dil != 1 || flops < CONV_MIN_FLOPS || !Native.use("conv", Lib.CUDNN)) {
            return false;
        }
        Cuda.conv2d(e, x, w, out, n, cin, h, wd, cout, r, pad, stride);
        // the bias is per output channel: a row bias over [cout, oh * ow] per image
        int rowEpilogue = ((epilogue & org.beehive.jitnn.kernels.GemmKernels.BIAS) != 0 ? org.beehive.jitnn.kernels.GemmKernels.ROW_BIAS : 0)
                | (epilogue & ~org.beehive.jitnn.kernels.GemmKernels.BIAS);
        epilogue(e, out, bias, cout, oh * ow, n, rowEpilogue);
        return true;
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
