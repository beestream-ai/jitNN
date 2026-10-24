package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.vectors.Half2;

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

    /**
     * y [n, hw, c] = x [n, c, hw]: channels-first to channels-last (also a weight [o, i, kh, kw] to
     * [o, kh, kw, i] with n = o, c = i, hw = kh kw); one work-item per element of y.
     */
    public static void toChannelsLast(KernelContext kc, FloatArray x, FloatArray y, int c, int hw, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = i % c;
            int p = (i / c) % hw;
            int b = i / (c * hw);
            y.set(i, x.get((b * c + ch) * hw + p));
        }
    }

    /**
     * out [n, c, hw] = epilogue(y [n, hw, c]): channels-last back to channels-first, with the
     * per-channel bias (BIAS) and ReLU of a convolution; one work-item per element of out.
     */
    public static void fromChannelsLast(KernelContext kc, FloatArray y, FloatArray out, FloatArray bias, int c, int hw, int total, int epilogue) {
        int i = kc.globalIdx;
        if (i < total) {
            int p = i % hw;
            int ch = (i / hw) % c;
            int b = i / (c * hw);
            float r = y.get((b * hw + p) * c + ch);
            if ((epilogue & GemmKernels.BIAS) != 0) {
                r = r + bias.get(ch);
            }
            if ((epilogue & GemmKernels.RELU) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            out.set(i, r);
        }
    }

    /** {@link #toChannelsLast} for FP16 arrays. */
    public static void toChannelsLastHalf(KernelContext kc, HalfFloatArray x, HalfFloatArray y, int c, int hw, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = i % c;
            int p = (i / c) % hw;
            int b = i / (c * hw);
            y.set(i, x.get((b * c + ch) * hw + p));
        }
    }

    /** {@link #toChannelsLast} from FP32 into FP16 (an FP32 activation entering an FP16 convolution). */
    public static void toChannelsLastToHalf(KernelContext kc, FloatArray x, HalfFloatArray y, int c, int hw, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = i % c;
            int p = (i / c) % hw;
            int b = i / (c * hw);
            y.set(i, new HalfFloat(x.get((b * c + ch) * hw + p)));
        }
    }

    /** {@link #fromChannelsLast} for FP16 arrays (FP32 bias, FP32 arithmetic). */
    public static void fromChannelsLastHalf(KernelContext kc, HalfFloatArray y, HalfFloatArray out, FloatArray bias, int c, int hw, int total, int epilogue) {
        int i = kc.globalIdx;
        if (i < total) {
            int p = i % hw;
            int ch = (i / hw) % c;
            int b = i / (c * hw);
            float r = y.get((b * hw + p) * c + ch).getFloat32();
            if ((epilogue & GemmKernels.BIAS) != 0) {
                r = r + bias.get(ch);
            }
            if ((epilogue & GemmKernels.RELU) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            out.set(i, new HalfFloat(r));
        }
    }

    /**
     * out = y + bias[c] + r (+ bias2[c] when hasBias2 != 0), then ReLU when relu != 0, on channels-last
     * tensors: the biases of the convolutions that produced y (and r), applied with the residual add.
     */
    public static void addBiasLast(KernelContext kc, FloatArray y, FloatArray r, FloatArray bias, FloatArray bias2, FloatArray out, int c, int total, int relu, int hasBias2) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = i % c;
            float v = y.get(i) + bias.get(ch) + r.get(i);
            if (hasBias2 != 0) {
                v = v + bias2.get(ch);
            }
            if (relu != 0) {
                v = v > 0.0f ? v : 0.0f;
            }
            out.set(i, v);
        }
    }

    /** {@link #addBiasLast} on FP16 tensors (FP32 bias, FP32 arithmetic). */
    public static void addBiasLastHalf(KernelContext kc, HalfFloatArray y, HalfFloatArray r, FloatArray bias, FloatArray bias2, HalfFloatArray out, int c, int total, int relu,
            int hasBias2) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = i % c;
            float v = y.get(i).getFloat32() + bias.get(ch) + r.get(i).getFloat32();
            if (hasBias2 != 0) {
                v = v + bias2.get(ch);
            }
            if (relu != 0) {
                v = v > 0.0f ? v : 0.0f;
            }
            out.set(i, new HalfFloat(v));
        }
    }

    /**
     * {@link #addBiasLastHalf} two channels per thread, with packed half2 loads and stores: for an even
     * channel count ({@code pairs} = total / 2).
     */
    public static void addBiasLastHalf2(KernelContext kc, HalfFloatArray y, HalfFloatArray r, FloatArray bias, FloatArray bias2, HalfFloatArray out, int c, int pairs, int relu,
            int hasBias2) {
        int p = kc.globalIdx;
        if (p < pairs) {
            int i = 2 * p;
            int ch = i % c;
            Half2 a = y.getHalf2(i);
            Half2 b = r.getHalf2(i);
            float v0 = a.getX().getFloat32() + bias.get(ch) + b.getX().getFloat32();
            float v1 = a.getY().getFloat32() + bias.get(ch + 1) + b.getY().getFloat32();
            if (hasBias2 != 0) {
                v0 = v0 + bias2.get(ch);
                v1 = v1 + bias2.get(ch + 1);
            }
            if (relu != 0) {
                v0 = v0 > 0.0f ? v0 : 0.0f;
                v1 = v1 > 0.0f ? v1 : 0.0f;
            }
            out.setHalf2(i, new Half2(new HalfFloat(v0), new HalfFloat(v1)));
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
