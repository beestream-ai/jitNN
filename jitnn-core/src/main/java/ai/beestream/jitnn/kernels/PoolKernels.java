package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * 2-D max and average pooling over NCHW (square window, symmetric stride and padding, floor
 * output size). Backward passes gather, one thread per input element, so overlapping windows need
 * no atomics.
 */
public final class PoolKernels {

    private PoolKernels() {
    }

    /** Max pooling; {@code arg} records the flat input index of each maximum for the backward pass. */
    public static void maxForward(KernelContext kc, FloatArray x, FloatArray out, IntArray arg, int h, int w, int oh, int ow, int k, int stride, int pad, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int owi = i % ow;
            int t = i / ow;
            int ohi = t % oh;
            int plane = t / oh;
            int base = plane * h * w;
            float best = Float.NEGATIVE_INFINITY;
            int bestAt = base;
            for (int kh = 0; kh < k; kh++) {
                int ih = ohi * stride - pad + kh;
                for (int kw = 0; kw < k; kw++) {
                    int iw = owi * stride - pad + kw;
                    if ((ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)) {
                        int at = base + ih * w + iw;
                        float v = x.get(at);
                        if (v > best) {
                            best = v;
                            bestAt = at;
                        }
                    }
                }
            }
            out.set(i, best);
            arg.set(i, bestAt);
        }
    }

    /**
     * {@link #maxForward} on [N, H, W, C] input and output, without the argmax (inference), then the
     * bias and ReLU of the convolution that produced x ({@code ep}: {@link GemmKernels#BIAS},
     * {@link GemmKernels#RELU}): per channel, they commute with the maximum.
     */
    public static void maxForwardLast(KernelContext kc, FloatArray x, FloatArray out, FloatArray bias, int c, int h, int w, int oh, int ow, int k, int stride, int pad, int total,
            int ep) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = i % c;
            int t = i / c;
            int owi = t % ow;
            t = t / ow;
            int ohi = t % oh;
            int n = t / oh;
            float best = Float.NEGATIVE_INFINITY;
            for (int kh = 0; kh < k; kh++) {
                int ih = ohi * stride - pad + kh;
                for (int kw = 0; kw < k; kw++) {
                    int iw = owi * stride - pad + kw;
                    if ((ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)) {
                        float v = x.get(((n * h + ih) * w + iw) * c + ch);
                        if (v > best) {
                            best = v;
                        }
                    }
                }
            }
            if ((ep & GemmKernels.BIAS) != 0) {
                best = best + bias.get(ch);
            }
            if ((ep & GemmKernels.RELU) != 0) {
                best = best > 0.0f ? best : 0.0f;
            }
            out.set(i, best);
        }
    }

    /** {@link #avgForward} on [N, H, W, C] input and output. */
    public static void avgForwardLast(KernelContext kc, FloatArray x, FloatArray out, int c, int h, int w, int oh, int ow, int k, int stride, int pad, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int ch = i % c;
            int t = i / c;
            int owi = t % ow;
            t = t / ow;
            int ohi = t % oh;
            int n = t / oh;
            float acc = 0.0f;
            for (int kh = 0; kh < k; kh++) {
                int ih = ohi * stride - pad + kh;
                for (int kw = 0; kw < k; kw++) {
                    int iw = owi * stride - pad + kw;
                    if ((ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)) {
                        acc += x.get(((n * h + ih) * w + iw) * c + ch);
                    }
                }
            }
            out.set(i, acc / (k * k));
        }
    }

    /** dx[e] = sum of dy over the windows whose maximum was element e. */
    public static void maxBackward(KernelContext kc, FloatArray dy, IntArray arg, FloatArray dx, int h, int w, int oh, int ow, int k, int stride, int pad, int total) {
        int e = kc.globalIdx;
        if (e < total) {
            int iw = e % w;
            int t = e / w;
            int ih = t % h;
            int plane = t / h;
            float acc = 0.0f;
            for (int kh = 0; kh < k; kh++) {
                int ohn = ih + pad - kh;
                int ohq = ohn / stride;
                for (int kw = 0; kw < k; kw++) {
                    int own = iw + pad - kw;
                    int owq = own / stride;
                    if ((ohn >= 0) & (own >= 0) & (ohq * stride == ohn) & (owq * stride == own) & (ohq < oh) & (owq < ow)) {
                        int o = (plane * oh + ohq) * ow + owq;
                        if (arg.get(o) == e) {
                            acc += dy.get(o);
                        }
                    }
                }
            }
            dx.set(e, acc);
        }
    }

    /** Average pooling, padding counted in the divisor (PyTorch's count_include_pad=True). */
    public static void avgForward(KernelContext kc, FloatArray x, FloatArray out, int h, int w, int oh, int ow, int k, int stride, int pad, int total) {
        int i = kc.globalIdx;
        if (i < total) {
            int owi = i % ow;
            int t = i / ow;
            int ohi = t % oh;
            int plane = t / oh;
            int base = plane * h * w;
            float acc = 0.0f;
            for (int kh = 0; kh < k; kh++) {
                int ih = ohi * stride - pad + kh;
                for (int kw = 0; kw < k; kw++) {
                    int iw = owi * stride - pad + kw;
                    if ((ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)) {
                        acc += x.get(base + ih * w + iw);
                    }
                }
            }
            out.set(i, acc / (k * k));
        }
    }

    public static void avgBackward(KernelContext kc, FloatArray dy, FloatArray dx, int h, int w, int oh, int ow, int k, int stride, int pad, int total) {
        int e = kc.globalIdx;
        if (e < total) {
            int iw = e % w;
            int t = e / w;
            int ih = t % h;
            int plane = t / h;
            float acc = 0.0f;
            for (int kh = 0; kh < k; kh++) {
                int ohn = ih + pad - kh;
                int ohq = ohn / stride;
                for (int kw = 0; kw < k; kw++) {
                    int own = iw + pad - kw;
                    int owq = own / stride;
                    if ((ohn >= 0) & (own >= 0) & (ohq * stride == ohn) & (owq * stride == own) & (ohq < oh) & (owq < ow)) {
                        acc += dy.get((plane * oh + ohq) * ow + owq);
                    }
                }
            }
            dx.set(e, acc / (k * k));
        }
    }
}
