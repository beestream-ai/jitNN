package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/** Small kernels around vendor-library calls: the epilogues and scalings the libraries do not fuse. */
public final class LibraryGlueKernels {

    private LibraryGlueKernels() {
    }

    /**
     * The GEMM epilogue (see {@link GemmKernels#BIAS}, {@code ROW_BIAS}, {@code RELU}, {@code GELU})
     * in place over {@code total / (rows * cols)} contiguous [rows, cols] matrices.
     */
    public static void epilogueBatched(KernelContext kc, FloatArray c, FloatArray bias, int rows, int cols, int total, int epilogue) {
        int i = kc.globalIdx;
        if (i < total) {
            int col = i % cols;
            int row = (i / cols) % rows;
            float r = c.get(i);
            if ((epilogue & GemmKernels.BIAS) != 0) {
                r = r + bias.get(col);
            }
            if ((epilogue & GemmKernels.ROW_BIAS) != 0) {
                r = r + bias.get(row);
            }
            if ((epilogue & GemmKernels.RELU) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            if ((epilogue & GemmKernels.GELU) != 0) {
                r = r * ElementwiseKernels.normalCdf(r);
            }
            if ((epilogue & GemmKernels.GELU_TANH) != 0) {
                r = ElementwiseKernels.geluTanh(r);
            }
            c.set(i, r);
        }
    }

    /** {@link #epilogueBatched} on an FP16 c (computed in FP32). */
    /** dst[i] = src[i] for i < n: a library operand moved into an array of exactly its size. */
    public static void copy(KernelContext kc, FloatArray src, FloatArray dst, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            dst.set(i, src.get(i));
        }
    }

    /** {@link #copy} for FP16 arrays. */
    public static void copyHalf(KernelContext kc, HalfFloatArray src, HalfFloatArray dst, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            dst.set(i, src.get(i));
        }
    }

    /** y = epilogue(float(x)): an FP16 product converted to FP32, then bias and activation (see GemmKernels). */
    public static void epilogueFromHalf(KernelContext kc, HalfFloatArray x, FloatArray bias, FloatArray y, int rows, int cols, int total, int epilogue) {
        int i = kc.globalIdx;
        if (i < total) {
            int col = i % cols;
            int row = (i / cols) % rows;
            float r = x.get(i).getFloat32();
            if ((epilogue & GemmKernels.BIAS) != 0) {
                r = r + bias.get(col);
            }
            if ((epilogue & GemmKernels.ROW_BIAS) != 0) {
                r = r + bias.get(row);
            }
            if ((epilogue & GemmKernels.RELU) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            if ((epilogue & GemmKernels.GELU) != 0) {
                r = r * ElementwiseKernels.normalCdf(r);
            }
            if ((epilogue & GemmKernels.GELU_TANH) != 0) {
                r = ElementwiseKernels.geluTanh(r);
            }
            y.set(i, r);
        }
    }

    public static void epilogueHalf(KernelContext kc, HalfFloatArray c, FloatArray bias, int rows, int cols, int total, int epilogue) {
        int i = kc.globalIdx;
        if (i < total) {
            int col = i % cols;
            int row = (i / cols) % rows;
            float r = c.get(i).getFloat32();
            if ((epilogue & GemmKernels.BIAS) != 0) {
                r = r + bias.get(col);
            }
            if ((epilogue & GemmKernels.ROW_BIAS) != 0) {
                r = r + bias.get(row);
            }
            if ((epilogue & GemmKernels.RELU) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            if ((epilogue & GemmKernels.GELU) != 0) {
                r = r * ElementwiseKernels.normalCdf(r);
            }
            if ((epilogue & GemmKernels.GELU_TANH) != 0) {
                r = ElementwiseKernels.geluTanh(r);
            }
            c.set(i, new HalfFloat(r));
        }
    }

    /** y[i] = half(epilogue(x[i])): an FP32 product with its bias and activation, stored as FP16. */
    public static void epilogueToHalf(KernelContext kc, FloatArray x, FloatArray bias, HalfFloatArray y, int rows, int cols, int total, int epilogue) {
        int i = kc.globalIdx;
        if (i < total) {
            int col = i % cols;
            float r = x.get(i);
            if ((epilogue & GemmKernels.BIAS) != 0) {
                r = r + bias.get(col);
            }
            if ((epilogue & GemmKernels.RELU) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            if ((epilogue & GemmKernels.GELU) != 0) {
                r = r * ElementwiseKernels.normalCdf(r);
            }
            if ((epilogue & GemmKernels.GELU_TANH) != 0) {
                r = ElementwiseKernels.geluTanh(r);
            }
            y.set(i, new HalfFloat(r));
        }
    }

    /** y[i] *= s for i < total. */
    public static void scale(KernelContext kc, FloatArray y, float s, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            y.set(i, y.get(i) * s);
        }
    }
}
