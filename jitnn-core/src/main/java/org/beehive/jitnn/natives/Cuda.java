package org.beehive.jitnn.natives;

import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Native;
import org.beehive.jitnn.engine.Native.Lib;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.CuBlasLt;
import uk.ac.manchester.tornado.cudnn.CuDnn;
import uk.ac.manchester.tornado.cufft.CuFft;
import uk.ac.manchester.tornado.cusparse.Cusparse;
import uk.ac.manchester.tornado.cutlass.Cutlass;

/**
 * cuBLAS, cuDNN, cuFFT and cuSPARSE calls as TornadoVM library tasks. jitNN's matrices are
 * row-major; cuBLAS is column-major, so a row-major C = op(A) op(B) is computed as the
 * column-major C^T = op(B)^T op(A)^T, which swaps the operands and keeps their transposes.
 */
final class Cuda {

    private Cuda() {
    }

    private static final int OP_N = 0;
    private static final int OP_T = 1;

    /** FP32 tensor cores (TF32) for cuBLAS products with {@code -Djitnn.native.tf32=true}; off by default (FP32 results). */
    private static final boolean TF32 = Boolean.getBoolean("jitnn.native.tf32");

    /** C [m, n] = op(A) op(B), row-major. */
    static void sgemm(Emitter e, FloatArray a, FloatArray b, FloatArray c, int m, int n, int k, int ta, int tb) {
        int opB = tb == 1 ? OP_T : OP_N;
        int opA = ta == 1 ? OP_T : OP_N;
        int ldb = tb == 1 ? k : n;
        int lda = ta == 1 ? m : k;
        if (TF32) {
            e.library(Lib.CUBLAS, "gemm", (g, id) -> g.libraryTask(id, CuBlas::cublasSgemmTF32, opB, opA, n, m, k, 1.0f, b, ldb, a, lda, 0.0f, c, n));
        } else {
            e.library(Lib.CUBLAS, "gemm", (g, id) -> g.libraryTask(id, CuBlas::cublasSgemm, opB, opA, n, m, k, 1.0f, b, ldb, a, lda, 0.0f, c, n));
        }
    }

    /** C[z] [m, n] = op(A[z]) op(B[z]) for z < batch, A[z] at z * strideA, B[z] at z * strideB (0: shared), C contiguous. */
    static void sgemmBatched(Emitter e, FloatArray a, FloatArray b, FloatArray c, int m, int n, int k, int ta, int tb, int batch, int strideA, int strideB) {
        int opB = tb == 1 ? OP_T : OP_N;
        int opA = ta == 1 ? OP_T : OP_N;
        int ldb = tb == 1 ? k : n;
        int lda = ta == 1 ? m : k;
        long sa = strideA;
        long sb = strideB;
        long sc = (long) m * n;
        e.library(Lib.CUBLAS, "gemm", (g, id) -> g.libraryTask(id, CuBlas::cublasSgemmStridedBatched, opB, opA, n, m, k, 1.0f, b, ldb, sb, a, lda, sa, 0.0f, c, n, sc, batch));
    }

    /** FP16 C [m, n] = op(A) op(B) for FP16 A and B. */
    static void gemmHalf(Emitter e, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int m, int n, int k, int ta, int tb) {
        int opB = tb == 1 ? OP_T : OP_N;
        int opA = ta == 1 ? OP_T : OP_N;
        int ldb = tb == 1 ? k : n;
        int lda = ta == 1 ? m : k;
        e.library(Lib.CUBLAS, "gemm", (g, id) -> g.libraryTask(id, CuBlas::cublasGemmExFP16, opB, opA, n, m, k, 1.0f, b, ldb, a, lda, 0.0f, c, n));
    }

    /** FP16 C [m, n] = A W^T + bias (+ ReLU, or tanh GELU) for FP16 A [m, k], W [n, k] and bias [n] (cuBLASLt epilogue). */
    static void gemmHalfBias(Emitter e, HalfFloatArray a, HalfFloatArray w, HalfFloatArray c, HalfFloatArray bias, int m, int n, int k, boolean relu, boolean gelu) {
        if (gelu) {
            e.library(Lib.CUBLAS, "gemmBias", (g, id) -> g.libraryTask(id, CuBlasLt::ltMatmulGeluBiasFP16, OP_T, OP_N, n, m, k, 1.0f, w, k, a, k, 0.0f, c, n, bias));
        } else if (relu) {
            e.library(Lib.CUBLAS, "gemmBias", (g, id) -> g.libraryTask(id, CuBlasLt::ltMatmulReluBiasFP16, OP_T, OP_N, n, m, k, 1.0f, w, k, a, k, 0.0f, c, n, bias));
        } else {
            e.library(Lib.CUBLAS, "gemmBias", (g, id) -> g.libraryTask(id, CuBlasLt::ltMatmulBiasFP16, OP_T, OP_N, n, m, k, 1.0f, w, k, a, k, 0.0f, c, n, bias));
        }
    }

    /** FP32 C [m, n] = op(A) op(B) for FP16 A and B (FP32 accumulation). */
    static void gemmF16(Emitter e, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k, int ta, int tb) {
        int opB = tb == 1 ? OP_T : OP_N;
        int opA = ta == 1 ? OP_T : OP_N;
        int ldb = tb == 1 ? k : n;
        int lda = ta == 1 ? m : k;
        e.library(Lib.CUBLAS, "gemm", (g, id) -> g.libraryTask(id, CuBlas::cublasGemmExFP16FP32, opB, opA, n, m, k, 1.0f, b, ldb, a, lda, 0.0f, c, n));
    }

    /**
     * Whether this TornadoVM has the library entry points jitNN uses when present: bfloat16 products
     * with an FP32 result, convolution gradients, and attention read from a packed projection. On an
     * older TornadoVM those operators keep jitNN's kernels.
     */
    static final boolean BF16_GEMM = has(CuBlas.class, "cublasGemmExBF16FP32");
    static final boolean CONV_BACKWARD = has(CuDnn.class, "cudnnConv2dBackwardData");
    static final boolean PACKED_ATTENTION = has(CuDnn.class, "sdpaForwardPackedQkv");

    private static boolean has(Class<?> type, String method) {
        for (var m : type.getMethods()) {
            if (m.getName().equals(method)) {
                return true;
            }
        }
        return false;
    }

    /** FP32 C [m, n] = op(A) op(B) for bfloat16 A and B (raw bits): one cuBLAS GemmEx on the tensor cores. */
    static void gemmBF16(Emitter e, ShortArray a, ShortArray b, FloatArray c, int m, int n, int k, int ta, int tb) {
        int opB = tb == 1 ? OP_T : OP_N;
        int opA = ta == 1 ? OP_T : OP_N;
        int ldb = tb == 1 ? k : n;
        int lda = ta == 1 ? m : k;
        e.library(Lib.CUBLAS, "gemmBF16", (g, id) -> g.libraryTask(id, CuBlas::cublasGemmExBF16FP32, opB, opA, n, m, k, 1.0f, b, ldb, a, lda, 0.0f, c, n));
    }

    /** The gradient of x of a cuDNN convolution (shapes as {@link #conv2d}): dx from w and dy. */
    static void conv2dBackwardData(Emitter e, FloatArray w, FloatArray dy, FloatArray dx, int n, int c, int h, int wd, int cout, int r, int pad, int stride) {
        e.library(Lib.CUDNN, "convBackward", (g, id) -> g.libraryTask(id, CuDnn::cudnnConv2dBackwardData, w, dy, dx, n, c, h, wd, cout, r, r, pad, stride));
    }

    /** The gradient of w of a cuDNN convolution (shapes as {@link #conv2d}): dw from x and dy. */
    static void conv2dBackwardFilter(Emitter e, FloatArray x, FloatArray dy, FloatArray dw, int n, int c, int h, int wd, int cout, int r, int pad, int stride) {
        e.library(Lib.CUDNN, "convBackward", (g, id) -> g.libraryTask(id, CuDnn::cudnnConv2dBackwardFilter, x, dy, dw, n, c, h, wd, cout, r, r, pad, stride));
    }

    /** cuDNN's fused FP16 attention on a packed projection [b, s, 3, h, d], O written as [b, s, h, d]. */
    static void attentionPacked(Emitter e, HalfFloatArray qkv, HalfFloatArray o, int b, int h, int s, int d, float scale, boolean causal) {
        e.library(Lib.CUDNN, "sdpa", (g, id) -> g.libraryTask(id, CuDnn::sdpaForwardPackedQkv, qkv, o, b, h, s, d, scale, causal));
    }

    /** cuDNN convolution: x [n, c, h, w] (NCHW), w [cout, c, r, r], one group, no dilation, into out [n, cout, oh, ow]. */
    static void conv2d(Emitter e, FloatArray x, FloatArray w, FloatArray out, int n, int c, int h, int wd, int cout, int r, int pad, int stride) {
        e.library(Lib.CUDNN, "conv", (g, id) -> g.libraryTask(id, CuDnn::cudnnConv2d, x, w, out, n, c, h, wd, cout, r, r, pad, stride));
    }

    /** cuDNN FP16 convolution (FP32 accumulation, tensor cores): as {@link #conv2d} on FP16 tensors. */
    static void conv2dHalf(Emitter e, HalfFloatArray x, HalfFloatArray w, HalfFloatArray out, int n, int c, int h, int wd, int cout, int r, int pad, int stride) {
        e.library(Lib.CUDNN, "convHalf", (g, id) -> g.libraryTask(id, CuDnn::cudnnConv2dFP16, x, w, out, n, c, h, wd, cout, r, r, pad, stride));
    }

    /** cuDNN channels-last convolution: x [n, h, w, c] (NHWC), w [cout, r, r, c] (KRSC), one group, no dilation, into out [n, oh, ow, cout]. */
    static void conv2dLast(Emitter e, FloatArray x, FloatArray w, FloatArray out, int n, int c, int h, int wd, int cout, int r, int pad, int stride) {
        e.library(Lib.CUDNN, "convNhwc", (g, id) -> g.libraryTask(id, CuDnn::cudnnConv2dNHWC, x, w, out, n, c, h, wd, cout, r, r, pad, stride));
    }

    /** {@link #conv2dLast} on FP16 tensors (FP32 accumulation, tensor cores). */
    static void conv2dLastHalf(Emitter e, HalfFloatArray x, HalfFloatArray w, HalfFloatArray out, int n, int c, int h, int wd, int cout, int r, int pad, int stride) {
        e.library(Lib.CUDNN, "convNhwcHalf", (g, id) -> g.libraryTask(id, CuDnn::cudnnConv2dNHWCFP16, x, w, out, n, c, h, wd, cout, r, r, pad, stride));
    }

    /** FP16 attention (cuDNN's fused SDPA, inference): q [b, h, t, d], k and v [b, h, s, d], all FP16. */
    static void attentionF16(Emitter e, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, HalfFloatArray o, int b, int h, int t, int s, int d, float scale, boolean causal) {
        e.library(Lib.CUDNN, "sdpa", (g, id) -> g.libraryTask(id, CuDnn::sdpaForward, q, k, v, o, b, h, t, s, d, scale, causal));
    }

    /** Row softmax of x [rows, cols] (cuDNN's accurate mode). */
    static void softmax(Emitter e, FloatArray x, FloatArray y, int rows, int cols) {
        e.library(Lib.CUDNN, "softmax", (g, id) -> g.libraryTask(id, CuDnn::cudnnSoftmax, x, y, rows, cols));
    }

    /** Complex-to-complex FFT of {@code batch} rows of n interleaved complex values; inverse is unnormalised. */
    static void fftC2C(Emitter e, FloatArray x, FloatArray y, int n, int batch, boolean inverse) {
        if (inverse) {
            e.library(Lib.CUFFT, "fft", (g, id) -> g.libraryTask(id, CuFft::cufftInverseC2C, x, y, n, batch));
        } else {
            e.library(Lib.CUFFT, "fft", (g, id) -> g.libraryTask(id, CuFft::cufftForwardC2C, x, y, n, batch));
        }
    }

    /** Real-to-complex FFT: {@code batch} rows of n reals into n / 2 + 1 complex values each. */
    static void fftR2C(Emitter e, FloatArray x, FloatArray y, int n, int batch) {
        e.library(Lib.CUFFT, "fft", (g, id) -> g.libraryTask(id, CuFft::cufftForwardR2C, x, y, n, batch));
    }

    /** C [rows, n] = A B for a CSR matrix A [rows, k] with nnz values and a dense row-major B [k, n]. */
    static void spmm(Emitter e, IntArray ptr, IntArray col, FloatArray val, FloatArray dense, FloatArray out, int rows, int k, int n, int nnz) {
        e.library(Lib.CUSPARSE, "spmm", (g, id) -> g.libraryTask(id, Cusparse::cusparseSpMM, rows, k, n, nnz, ptr, col, val, dense, out));
    }

    /** FP16 C[z] [m, n] = A[z] [m, k] B[z] [k, n] for z < batch, all contiguous (CUTLASS; n and k multiples of 4). */
    static void hgemmBatched(Emitter e, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int m, int n, int k, int batch) {
        e.library(Lib.CUTLASS, "gemm", (g, id) -> g.libraryTask(id, Cutlass::cutlassHgemmBatched, m, n, k, 1.0f, a, b, 0.0f, c, batch));
    }

    /** Runs cuDNN's fused FP16 attention once on a small problem: null if it builds a plan and gives finite results. */
    static String probeAttention() {
        int b = 2, h = 4, t = 128, d = 64;
        int n = b * h * t * d;
        HalfFloatArray q = new HalfFloatArray(n);
        HalfFloatArray k = new HalfFloatArray(n);
        HalfFloatArray v = new HalfFloatArray(n);
        for (int i = 0; i < n; i++) {
            q.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat((i % 7) * 0.1f));
            k.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat((i % 5) * 0.1f));
            v.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat((i % 3) * 0.5f));
        }
        HalfFloatArray o = new HalfFloatArray(n);
        TaskGraph g = Native.probeGraph("SDPA").transferToDevice(DataTransferMode.EVERY_EXECUTION, q, k, v)
                .libraryTask("t", CuDnn::sdpaForward, q, k, v, o, b, h, t, t, d, 0.125f, false).transferToHost(DataTransferMode.EVERY_EXECUTION, o);
        Native.runProbe(g);
        // v's rows repeat 0, 0.5, 1.0 down the keys, so every output lies in [0, 1]
        for (int i = 0; i < n; i++) {
            float x = o.get(i).getFloat32();
            if (!(x >= -1e-3f && x <= 1.001f)) {
                return "probe returned " + x;
            }
        }
        return null;
    }

    /**
     * One small FP16 convolution: null when it works (a TornadoVM without the FP16 convolution fails
     * here, as does a cuDNN that has no FP16 algorithm for the device).
     */
    static String probeConvHalf() {
        int c = 8, h = 8, k = 8;
        HalfFloatArray x = new HalfFloatArray(c * h * h);
        HalfFloatArray w = new HalfFloatArray(k * c * 9);
        x.init(new uk.ac.manchester.tornado.api.types.HalfFloat(1.0f));
        w.init(new uk.ac.manchester.tornado.api.types.HalfFloat(0.125f));
        HalfFloatArray o = new HalfFloatArray(k * h * h);
        TaskGraph g = Native.probeGraph("ConvHalf").transferToDevice(DataTransferMode.EVERY_EXECUTION, x, w)
                .libraryTask("t", CuDnn::cudnnConv2dFP16, x, w, o, 1, c, h, h, k, 3, 3, 1, 1).transferToHost(DataTransferMode.EVERY_EXECUTION, o);
        Native.runProbe(g);
        // an interior output sums 8 channels x 9 taps of 0.125
        float centre = o.get(h * h / 2 + h / 2).getFloat32();
        return Math.abs(centre - 9.0f) < 1e-2f ? null : "probe returned " + centre;
    }

    /**
     * One small channels-last convolution, FP16 when {@code half}: null when it works (a TornadoVM
     * without the NHWC convolutions fails here).
     */
    static String probeConvLast(boolean half) {
        int c = 8, h = 8, k = 8;
        TaskGraph g = Native.probeGraph(half ? "ConvNhwcHalf" : "ConvNhwc");
        float centre;
        if (half) {
            HalfFloatArray x = new HalfFloatArray(c * h * h);
            HalfFloatArray w = new HalfFloatArray(k * c * 9);
            x.init(new uk.ac.manchester.tornado.api.types.HalfFloat(1.0f));
            w.init(new uk.ac.manchester.tornado.api.types.HalfFloat(0.125f));
            HalfFloatArray o = new HalfFloatArray(k * h * h);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, x, w).libraryTask("t", CuDnn::cudnnConv2dNHWCFP16, x, w, o, 1, c, h, h, k, 3, 3, 1, 1)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, o);
            Native.runProbe(g);
            centre = o.get((h * h / 2 + h / 2) * k).getFloat32();
        } else {
            FloatArray x = new FloatArray(c * h * h);
            FloatArray w = new FloatArray(k * c * 9);
            x.init(1.0f);
            w.init(0.125f);
            FloatArray o = new FloatArray(k * h * h);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, x, w).libraryTask("t", CuDnn::cudnnConv2dNHWC, x, w, o, 1, c, h, h, k, 3, 3, 1, 1)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, o);
            Native.runProbe(g);
            centre = o.get((h * h / 2 + h / 2) * k);
        }
        // an interior output sums 8 channels x 9 taps of 0.125
        return Math.abs(centre - 9.0f) < 1e-2f ? null : "probe returned " + centre;
    }

    /** Runs one small call of {@code lib} on the current device: null if it works, otherwise why not. */
    static String probe(Lib lib) {
        // a missing native library (libcublas, libcudnn, ...) fails the probe's first execution
        TaskGraph g = Native.probeGraph(lib.name());
        float[] expected;
        FloatArray out;
        switch (lib) {
            case CUBLAS -> {
                // [1 2; 3 4] [5 6; 7 8] = [19 22; 43 50]
                FloatArray a = FloatArray.fromElements(1, 2, 3, 4);
                FloatArray b = FloatArray.fromElements(5, 6, 7, 8);
                out = new FloatArray(4);
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b).libraryTask("t", CuBlas::cublasSgemm, OP_N, OP_N, 2, 2, 2, 1.0f, b, 2, a, 2, 0.0f, out, 2);
                expected = new float[] { 19, 22, 43, 50 };
            }
            case CUDNN -> {
                FloatArray x = FloatArray.fromElements(0, 0, 0, 0);
                out = new FloatArray(4);
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, x).libraryTask("t", CuDnn::cudnnSoftmax, x, out, 1, 4);
                expected = new float[] { 0.25f, 0.25f, 0.25f, 0.25f };
            }
            case CUFFT -> {
                // FFT of [1, 0, 0, 0] (complex) is all ones
                FloatArray x = FloatArray.fromElements(1, 0, 0, 0, 0, 0, 0, 0);
                out = new FloatArray(8);
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, x).libraryTask("t", CuFft::cufftForwardC2C, x, out, 4, 1);
                expected = new float[] { 1, 0, 1, 0, 1, 0, 1, 0 };
            }
            case NCCL -> {
                // no probe: a communicator needs every rank's device, and NcclCollective builds it
                return null;
            }
            case CUTLASS -> {
                // a 4 x 4 identity times B gives B, in FP16
                HalfFloatArray a = new HalfFloatArray(16);
                HalfFloatArray b = new HalfFloatArray(16);
                for (int i = 0; i < 16; i++) {
                    a.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat(i % 5 == 0 ? 1 : 0));
                    b.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat(i));
                }
                HalfFloatArray c = new HalfFloatArray(16);
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b).libraryTask("t", Cutlass::cutlassHgemm, 4, 4, 4, 1.0f, a, b, 0.0f, c).transferToHost(DataTransferMode.EVERY_EXECUTION,
                        c);
                Native.runProbe(g);
                for (int i = 0; i < 16; i++) {
                    if (c.get(i).getFloat32() != i) {
                        return "probe returned " + c.get(i).getFloat32() + " for " + i;
                    }
                }
                return null;
            }
            default -> {
                // [[2, 0], [0, 3]] (CSR) times [[1], [1]]
                IntArray ptr = IntArray.fromElements(0, 1, 2);
                IntArray col = IntArray.fromElements(0, 1);
                FloatArray val = FloatArray.fromElements(2, 3);
                FloatArray dense = FloatArray.fromElements(1, 1);
                out = new FloatArray(2);
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, ptr, col, val, dense).libraryTask("t", Cusparse::cusparseSpMM, 2, 2, 1, 2, ptr, col, val, dense, out);
                expected = new float[] { 2, 3 };
            }
        }
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        Native.runProbe(g);
        for (int i = 0; i < expected.length; i++) {
            if (Math.abs(out.get(i) - expected[i]) > 1e-4f) {
                return "probe returned " + out.get(i) + " for " + expected[i];
            }
        }
        return null;
    }
}
