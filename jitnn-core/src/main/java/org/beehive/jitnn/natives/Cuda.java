package org.beehive.jitnn.natives;

import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Native;
import org.beehive.jitnn.engine.Native.Lib;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cudnn.CuDnn;
import uk.ac.manchester.tornado.cufft.CuFft;
import uk.ac.manchester.tornado.cusparse.Cusparse;

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

    /** FP32 C [m, n] = op(A) op(B) for FP16 A and B (FP32 accumulation). */
    static void gemmF16(Emitter e, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k, int ta, int tb) {
        int opB = tb == 1 ? OP_T : OP_N;
        int opA = ta == 1 ? OP_T : OP_N;
        int ldb = tb == 1 ? k : n;
        int lda = ta == 1 ? m : k;
        e.library(Lib.CUBLAS, "gemm", (g, id) -> g.libraryTask(id, CuBlas::cublasGemmExFP16FP32, opB, opA, n, m, k, 1.0f, b, ldb, a, lda, 0.0f, c, n));
    }

    /** cuDNN convolution: x [n, c, h, w] (NCHW), w [cout, c, r, r], one group, no dilation, into out [n, cout, oh, ow]. */
    static void conv2d(Emitter e, FloatArray x, FloatArray w, FloatArray out, int n, int c, int h, int wd, int cout, int r, int pad, int stride) {
        e.library(Lib.CUDNN, "conv", (g, id) -> g.libraryTask(id, CuDnn::cudnnConv2d, x, w, out, n, c, h, wd, cout, r, r, pad, stride));
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
