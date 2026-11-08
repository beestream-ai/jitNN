/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn;

import ai.beestream.jitnn.ops.AttentionSoftmax;
import ai.beestream.jitnn.ops.BatchNorm2d;
import ai.beestream.jitnn.ops.Binary;
import ai.beestream.jitnn.ops.Conv2d;
import ai.beestream.jitnn.ops.ConvTranspose2d;
import ai.beestream.jitnn.ops.CrossEntropy;
import ai.beestream.jitnn.ops.Dropout;
import ai.beestream.jitnn.ops.Embedding;
import ai.beestream.jitnn.ops.FlashAttention;
import ai.beestream.jitnn.ops.Linear;
import ai.beestream.jitnn.ops.Nll;
import ai.beestream.jitnn.ops.Pool2d;
import ai.beestream.jitnn.ops.Recurrent;
import ai.beestream.jitnn.ops.Rope;
import ai.beestream.jitnn.ops.Sampling;
import ai.beestream.jitnn.ops.RowNorm;
import ai.beestream.jitnn.ops.Upsample2d;
import ai.beestream.jitnn.kernels.ResizeKernels;

/** Functional operators that are not methods of {@link Tensor}. */
public final class F {

    /** Fused attention kernels for FP32 attention (default); {@code -Djitnn.fusedAttention=false} composes GEMMs and softmax instead. */
    private static final boolean FUSED_ATTENTION = JitnnConfig.current().get(JitnnConfig.FUSED_ATTENTION);

    private F() {
    }

    /**
     * y = x W^T + b, with W stored [out, in] and b of shape [out] (b may be null); one fused GEMM.
     * x may have any leading dimensions: [..., in] gives [..., out].
     */
    public static Tensor linear(Tensor x, Tensor weight, Tensor bias) {
        return linear(x, weight, bias, false);
    }

    /** relu(x W^T + b), with the bias and ReLU fused into the GEMM epilogue. */
    public static Tensor linearRelu(Tensor x, Tensor weight, Tensor bias) {
        return linear(x, weight, bias, true);
    }

    private static Tensor linear(Tensor x, Tensor weight, Tensor bias, boolean relu) {
        Tensor x2 = x.rank() <= 2 ? x : x.reshape(-1, x.dim(-1));
        Tensor y;
        if (Autocast.current() == DType.F16 && !Grad.isEnabled() && !Autocast.halfActivations() && x2.dtype() == DType.F32 && weight.dtype() == DType.F32
                && Linear.readsFp32Input(x2.dim(0), weight.dim(0), x2.dim(1))) {
            // FP16 inference on the tensor cores: the FP32 input is rounded as the kernel stages it
            y = Linear.apply(x2, Autocast.weight(weight), bias, relu);
        } else {
            // half-precision inference: the bias too is cast once per version (FP16 for cuBLASLt's epilogue)
            Tensor b = bias != null && Autocast.halfActivations() ? Autocast.weight(bias) : bias;
            y = Linear.apply(Autocast.operand(x2), Autocast.weight(weight), b, relu);
        }
        if (x.rank() <= 2) {
            return y;
        }
        int[] dims = x.shape().dims();
        dims[dims.length - 1] = weight.dim(0);
        return y.reshape(dims);
    }

    /** Per-sample negative log-likelihood of INT32 {@code labels} under log-probabilities {@code logp}. */
    public static Tensor nll(Tensor logp, Tensor labels) {
        return Nll.apply(logp, labels);
    }

    /** Mean cross-entropy between {@code logits} [N, C] and INT32 class labels [N] (fused kernels). */
    public static Tensor crossEntropy(Tensor logits, Tensor labels) {
        return CrossEntropy.apply(logits, labels);
    }

    /** 2-D convolution (NCHW), square kernel; bias may be null. */
    public static Tensor conv2d(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return Conv2d.apply(x, convWeight(x, weight, stride, padding, dilation, groups), bias, stride, padding, dilation, groups, false);
    }

    /** The weight of a convolution in half-precision inference cast to FP16 once per version, as for linear layers. */
    private static Tensor convWeight(Tensor x, Tensor weight, int stride, int padding, int dilation, int groups) {
        return Conv2d.halfPath(x, weight, stride, padding, dilation, groups) ? Autocast.weight(weight) : weight;
    }

    /** relu(conv2d(...)) with the bias and ReLU fused into the convolution's output. */
    public static Tensor conv2dRelu(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return Conv2d.apply(x, convWeight(x, weight, stride, padding, dilation, groups), bias, stride, padding, dilation, groups, true);
    }

    /**
     * Batch normalisation of an NCHW (or NC) tensor. In training it uses the batch statistics and
     * updates {@code runningMean}/{@code runningVar} (persistent tensors) in place.
     */
    public static Tensor batchNorm(Tensor x, Tensor runningMean, Tensor runningVar, Tensor weight, Tensor bias, boolean training, float momentum, float eps) {
        return BatchNorm2d.apply(x, weight, bias, runningMean, runningVar, training, eps, momentum, false);
    }

    /** relu(batchNorm(...)), fused. */
    public static Tensor batchNormRelu(Tensor x, Tensor runningMean, Tensor runningVar, Tensor weight, Tensor bias, boolean training, float momentum, float eps) {
        return BatchNorm2d.apply(x, weight, bias, runningMean, runningVar, training, eps, momentum, true);
    }

    /**
     * Transposed 2-D convolution (NCHW), PyTorch's conv_transpose2d: weight [Cin, Cout/groups, K, K],
     * square kernel; bias may be null.
     */
    public static Tensor convTranspose2d(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int outputPadding, int dilation, int groups) {
        return ConvTranspose2d.apply(x, weight, bias, stride, padding, outputPadding, dilation, groups);
    }

    /** Nearest-neighbour upsampling of x [N, C, H, W] by {@code scale} (PyTorch's mode='nearest'). */
    public static Tensor upsampleNearest(Tensor x, float scale) {
        return Upsample2d.scale(x, scale, scale, ResizeKernels.NEAREST, false);
    }

    /** Bilinear upsampling of x [N, C, H, W] by {@code scale}. */
    public static Tensor upsampleBilinear(Tensor x, float scale, boolean alignCorners) {
        return Upsample2d.scale(x, scale, scale, ResizeKernels.BILINEAR, alignCorners);
    }

    /** Resizes x [N, C, H, W] to [N, C, height, width], nearest or bilinear (PyTorch's interpolate with size). */
    public static Tensor interpolate(Tensor x, int height, int width, boolean bilinear, boolean alignCorners) {
        return Upsample2d.size(x, height, width, bilinear ? ResizeKernels.BILINEAR : ResizeKernels.NEAREST, alignCorners);
    }

    /**
     * N-dimensional convolution of x [N, C, ...] (1 or 3 spatial axes; 2-D has {@link #conv2d}) with
     * weight [Cout, C, ...kernel], groups 1: the windows are unfolded into columns and multiplied by
     * the weight in one batched GEMM. Bias may be null.
     */
    private static Tensor convNd(Tensor x, Tensor weight, Tensor bias, int[] stride, int[] pad, int[] dil, int groups) {
        int r = x.rank() - 2;
        int[] kernel = java.util.Arrays.copyOfRange(weight.shape().dims(), 2, 2 + r);
        int cout = weight.dim(0);
        if (weight.rank() != r + 2 || groups < 1 || x.dim(1) % groups != 0 || cout % groups != 0 || weight.dim(1) * groups != x.dim(1)) {
            throw new IllegalArgumentException("convolution: input " + x.shape() + ", weight " + weight.shape() + " and groups " + groups + " do not match");
        }
        // columns [N, C * K, L], rows channel-major, so each group's rows are contiguous
        Tensor cols = ai.beestream.jitnn.ops.Spatial.unfold(x, kernel, stride, pad, dil);
        Tensor y;
        if (groups == 1) {
            y = weight.reshape(cout, -1).matmul(cols);
        } else {
            int rows = cols.dim(1) / groups;
            int n = cols.dim(0);
            // [N, G, Cout / G, rows] x [N, G, rows, L] -> [N, G, Cout / G, L] (the batched product needs equal batch dimensions)
            Tensor w = weight.reshape(1, groups, cout / groups, rows).expand(Shape.of(n, groups, cout / groups, rows));
            y = w.matmul(cols.reshape(n, groups, rows, cols.dim(2))).reshape(n, cout, cols.dim(2));
        }
        if (bias != null) {
            y = y.add(bias.reshape(1, cout, 1));
        }
        int[] dims = new int[r + 2];
        dims[0] = x.dim(0);
        dims[1] = cout;
        for (int k = 0; k < r; k++) {
            dims[2 + k] = (x.dim(2 + k) + 2 * pad[k] - dil[k] * (kernel[k] - 1) - 1) / stride[k] + 1;
        }
        return y.reshape(dims);
    }

    /** 1-D convolution of x [N, C, L] with weight [Cout, C, K]; bias may be null. */
    public static Tensor conv1d(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int dilation) {
        return conv1d(x, weight, bias, stride, padding, dilation, 1);
    }

    /** Grouped 1-D convolution: weight [Cout, C / groups, K]; groups = C is depthwise. */
    public static Tensor conv1d(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return convNd(x, weight, bias, new int[] { stride }, new int[] { padding }, new int[] { dilation }, groups);
    }

    /** 3-D convolution of x [N, C, D, H, W] with weight [Cout, C, KD, KH, KW]; bias may be null. */
    public static Tensor conv3d(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int dilation) {
        return conv3d(x, weight, bias, stride, padding, dilation, 1);
    }

    /** Grouped 3-D convolution: weight [Cout, C / groups, KD, KH, KW]. */
    public static Tensor conv3d(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return convNd(x, weight, bias, new int[] { stride, stride, stride }, new int[] { padding, padding, padding }, new int[] { dilation, dilation, dilation }, groups);
    }

    /**
     * Convolution over 1 to 3 spatial axes with a size per axis: {@code pads} holds the padding
     * before every axis, then after every axis (ONNX's order), so it may be asymmetric. Square
     * symmetric 2-D cases take {@link #conv2d}'s kernels; the rest the unfold (im2col) path.
     */
    public static Tensor conv(Tensor x, Tensor weight, Tensor bias, int[] stride, int[] pads, int[] dilation, int groups) {
        int r = x.rank() - 2;
        int[] before = java.util.Arrays.copyOfRange(pads, 0, r);
        int[] after = java.util.Arrays.copyOfRange(pads, r, 2 * r);
        boolean symmetric = java.util.Arrays.equals(before, after);
        if (r == 2 && symmetric && uniform(stride) && uniform(before) && uniform(dilation) && weight.dim(2) == weight.dim(3)) {
            return conv2d(x, weight, bias, stride[0], before[0], dilation[0], groups);
        }
        if (!symmetric) {
            x = pad(x, before, after, PadMode.CONSTANT, 0.0f);
            before = new int[r];
        }
        return convNd(x, weight, bias, stride, before, dilation, groups);
    }

    private static boolean uniform(int[] v) {
        for (int x : v) {
            if (x != v[0]) {
                return false;
            }
        }
        return true;
    }

    /** How {@link #pool} combines a window. */
    public enum PoolMode {
        MAX, AVG, LP
    }

    /**
     * Pooling over 1 to 3 spatial axes of x [N, C, ...] with a size per axis, as ONNX's MaxPool,
     * AveragePool and LpPool and PyTorch's pooling layers: {@code pads} holds the padding before
     * every axis, then after every axis; {@code ceil} rounds the output size up (a window must
     * still start inside the input or its leading padding); average pooling counts the padding
     * when {@code countIncludePad} (never the positions added by {@code ceil}); LP pooling
     * computes (sum |x|^p)^(1/p).
     */
    public static Tensor pool(Tensor x, PoolMode mode, int[] kernel, int[] stride, int[] pads, int[] dilation, boolean ceil, boolean countIncludePad, float p) {
        int r = x.rank() - 2;
        int n = x.dim(0);
        int c = x.dim(1);
        int[] in = new int[r];
        int[] out = new int[r];
        int[] before = new int[r];
        int[] after = new int[r];
        for (int a = 0; a < r; a++) {
            in[a] = x.dim(2 + a);
            int span = dilation[a] * (kernel[a] - 1) + 1;
            int room = in[a] + pads[a] + pads[r + a] - span;
            if (room < 0) {
                throw new IllegalArgumentException("pooling window " + java.util.Arrays.toString(kernel) + " larger than the padded input " + x.shape());
            }
            out[a] = (ceil ? (room + stride[a] - 1) / stride[a] : room / stride[a]) + 1;
            if (ceil && (out[a] - 1) * stride[a] >= in[a] + pads[a]) {
                out[a]--;
            }
            before[a] = pads[a];
            // the windows that ceil mode adds read past the end: padding nobody counts
            after[a] = Math.max(pads[r + a], (out[a] - 1) * stride[a] + span - in[a] - pads[a]);
        }
        float fill = mode == PoolMode.MAX ? -3.4028235e38f : 0.0f;
        Tensor planes = x.reshape(spatialOf(x, n * c));
        Tensor xp = pad(planes, before, after, PadMode.CONSTANT, fill);
        int[] zero = new int[r];
        Tensor cols = ai.beestream.jitnn.ops.Spatial.unfold(xp, kernel, stride, zero, dilation);
        Tensor y = switch (mode) {
            case MAX -> cols.max(1, false);
            case LP -> p == 2.0f ? cols.square().sum(1, false).sqrt() : cols.abs().pow(p).sum(1, false).pow(1.0f / p);
            case AVG -> {
                // the divisor of every window, known from the shapes alone: computed on the host
                int windows = 1;
                for (int v : out) {
                    windows *= v;
                }
                float[] count = new float[windows];
                int[] at = new int[r];
                for (int w = 0; w < windows; w++) {
                    int rest = w;
                    for (int a = r - 1; a >= 0; a--) {
                        at[a] = rest % out[a];
                        rest /= out[a];
                    }
                    long cnt = 1;
                    for (int a = 0; a < r; a++) {
                        int lo = countIncludePad ? -pads[a] : 0;
                        int hi = countIncludePad ? in[a] + pads[r + a] : in[a];
                        int inside = 0;
                        for (int k = 0; k < kernel[a]; k++) {
                            int pos = at[a] * stride[a] - pads[a] + k * dilation[a];
                            inside += pos >= lo && pos < hi ? 1 : 0;
                        }
                        cnt *= inside;
                    }
                    count[w] = 1.0f / Math.max(cnt, 1);
                }
                yield cols.sum(1, false).mul(Tensor.of(count, 1, windows));
            }
        };
        int[] dims = new int[r + 2];
        dims[0] = n;
        dims[1] = c;
        System.arraycopy(out, 0, dims, 2, r);
        return y.reshape(dims);
    }

    /** Max or average pooling over 1 or 3 spatial axes (2-D has {@link #maxPool2d}); average counts padding. */
    private static Tensor poolNd(Tensor x, int kernel, int stride, int padding, boolean max) {
        int r = x.rank() - 2;
        int[] k = new int[r];
        int[] s = new int[r];
        int[] p = new int[r];
        int[] d = new int[r];
        java.util.Arrays.fill(k, kernel);
        java.util.Arrays.fill(s, stride);
        java.util.Arrays.fill(p, padding);
        java.util.Arrays.fill(d, 1);
        int n = x.dim(0);
        int c = x.dim(1);
        Tensor cols = ai.beestream.jitnn.ops.Spatial.unfold(x.reshape(spatialOf(x, n * c)), k, s, p, d);
        int windows = cols.dim(1);
        Tensor pooled = max ? padLarge(cols, x, k, s, p).max(1, false) : cols.sum(1, false).mul(1.0f / windows);
        int[] dims = x.shape().dims();
        for (int a = 0; a < r; a++) {
            dims[2 + a] = (x.dim(2 + a) + 2 * padding - kernel) / stride + 1;
        }
        return pooled.reshape(dims);
    }

    /** x [N, C, ...spatial] viewed as [N C, 1, ...spatial]. */
    private static int[] spatialOf(Tensor x, int planes) {
        int[] dims = x.shape().dims();
        dims[0] = planes;
        dims[1] = 1;
        return dims;
    }

    /** Max pooling pads with -inf (PyTorch), not 0: the padded positions are lowered before the max. */
    private static Tensor padLarge(Tensor cols, Tensor x, int[] k, int[] s, int[] p) {
        boolean padded = false;
        for (int v : p) {
            padded |= v > 0;
        }
        if (!padded) {
            return cols;
        }
        int[] dims = x.shape().dims();
        dims[0] = 1;
        dims[1] = 1;
        int[] ones = new int[k.length];
        java.util.Arrays.fill(ones, 1);
        Tensor inside = ai.beestream.jitnn.ops.Spatial.unfold(Tensor.ones(dims), k, s, p, ones).reshape(cols.dim(1), cols.dim(2));
        return where(inside, cols, Tensor.scalar(-3.4028235e38f));
    }

    /**
     * Max pooling of x [N, C, L] over windows of {@code kernel} with the given stride and padding (padding
     * never wins), PyTorch's max_pool1d.
     */
    public static Tensor maxPool1d(Tensor x, int kernel, int stride, int padding) {
        return poolNd(x, kernel, stride, padding, true);
    }

    /**
     * Average pooling of x [N, C, L], PyTorch's avg_pool1d; padding counts in the divisor, as PyTorch's
     * default.
     */
    public static Tensor avgPool1d(Tensor x, int kernel, int stride, int padding) {
        return poolNd(x, kernel, stride, padding, false);
    }

    /**
     * Max pooling of x [N, C, D, H, W] over cubic windows of {@code kernel} (padding never wins), PyTorch's
     * max_pool3d.
     */
    public static Tensor maxPool3d(Tensor x, int kernel, int stride, int padding) {
        return poolNd(x, kernel, stride, padding, true);
    }

    /**
     * Average pooling of x [N, C, D, H, W] over cubic windows, PyTorch's avg_pool3d; padding counts in the
     * divisor.
     */
    public static Tensor avgPool3d(Tensor x, int kernel, int stride, int padding) {
        return poolNd(x, kernel, stride, padding, false);
    }

    /** Adaptive average pooling of x [N, C, ...] (1 to 3 spatial axes) to the given output sizes (PyTorch's bins). */
    public static Tensor adaptiveAvgPool(Tensor x, int... out) {
        return ai.beestream.jitnn.ops.Spatial.adaptivePool(x, out, false);
    }

    /** Adaptive max pooling of x [N, C, ...] (1 to 3 spatial axes) to the given output sizes. */
    public static Tensor adaptiveMaxPool(Tensor x, int... out) {
        return ai.beestream.jitnn.ops.Spatial.adaptivePool(x, out, true);
    }

    /** How {@link #pad} fills: a constant, the reflection without the edge, or the edge replicated. */
    public enum PadMode {
        /** Fills with a constant value. */
        CONSTANT,
        /** Mirrors the input about its edge, without repeating the edge element. */
        REFLECT,
        /** Repeats the edge element. */
        REPLICATE
    }

    /**
     * Pads the spatial axes of x [N, C, ...]: {@code before[k]} and {@code after[k]} elements on
     * spatial axis k, PyTorch's F.pad for constant, reflect and replicate modes.
     */
    public static Tensor pad(Tensor x, int[] before, int[] after, PadMode mode, float value) {
        return ai.beestream.jitnn.ops.Spatial.pad(x, before, after, mode.ordinal(), value);
    }

    /**
     * Max pooling of x [N, C, H, W] over square windows of {@code kernel} with the given stride and padding,
     * PyTorch's max_pool2d.
     */
    public static Tensor maxPool2d(Tensor x, int kernel, int stride, int padding) {
        return Pool2d.max(x, kernel, stride, padding);
    }

    /**
     * Average pooling of x [N, C, H, W] over square windows, PyTorch's avg_pool2d; padding counts in the
     * divisor, as PyTorch's default.
     */
    public static Tensor avgPool2d(Tensor x, int kernel, int stride, int padding) {
        return Pool2d.avg(x, kernel, stride, padding);
    }

    /** Mean over the spatial dimensions: [N, C, H, W] to [N, C]. */
    public static Tensor globalAvgPool2d(Tensor x) {
        return x.reshape(x.dim(0), x.dim(1), -1).mean(2, false);
    }

    /** Inverted dropout with a fresh mask per step: {@code seed} is an INT32 [2] host tensor. */
    public static Tensor dropout(Tensor x, float p, Tensor seed, int stream) {
        return Dropout.apply(x, p, seed, stream);
    }

    /** LayerNorm over the last axis: (x - mean) / sqrt(var + eps) * weight + bias (bias may be null). */
    public static Tensor layerNorm(Tensor x, Tensor weight, Tensor bias, float eps) {
        return RowNorm.layerNorm(x, weight, bias, eps);
    }

    /**
     * GroupNorm (Wu and He) of x [N, C, ...] in {@code groups} groups of channels: each (sample,
     * group) normalised over its channels and positions, then a per-channel affine (weight and bias
     * may be null, as PyTorch with affine=False).
     */
    public static Tensor groupNorm(Tensor x, int groups, Tensor weight, Tensor bias, float eps) {
        int n = x.dim(0);
        int c = x.dim(1);
        if (c % groups != 0) {
            throw new IllegalArgumentException(c + " channels do not split into " + groups + " groups");
        }
        int cols = x.numel() / (n * groups);
        Tensor y = RowNorm.layerNorm(x.reshape(n * groups, cols), Tensor.ones(cols), null, eps).reshape(n, c, -1);
        if (weight != null) {
            y = y.mul(weight.reshape(1, c, 1));
        }
        if (bias != null) {
            y = y.add(bias.reshape(1, c, 1));
        }
        return y.reshape(x.shape().dims());
    }

    /** InstanceNorm: GroupNorm with one channel per group (affine when weight and bias are given). */
    public static Tensor instanceNorm(Tensor x, Tensor weight, Tensor bias, float eps) {
        return groupNorm(x, x.dim(1), weight, bias, eps);
    }

    /** Local response normalisation across channels, PyTorch's local_response_norm (k 1, alpha 1e-4, beta 0.75 by default). */
    public static Tensor localResponseNorm(Tensor x, int size, float alpha, float beta, float k) {
        return ai.beestream.jitnn.ops.Lrn.apply(x, size, alpha, beta, k);
    }

    /**
     * Weight normalisation (Salimans and Kingma): w = g v / ||v|| with the norm over every axis but
     * 0, g of shape [out] (PyTorch's weight_norm with dim 0).
     */
    public static Tensor weightNorm(Tensor v, Tensor g) {
        int out = v.dim(0);
        Tensor norm = v.reshape(out, -1).square().sum(1, false).sqrt();
        int[] dims = new int[v.rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = out;
        return v.mul(g.reshape(out).div(norm).reshape(dims));
    }

    /** RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * weight. */
    public static Tensor rmsNorm(Tensor x, Tensor weight, float eps) {
        return RowNorm.rmsNorm(x, weight, eps);
    }

    /** Rows of {@code weight} [V, D] picked by the INT32 {@code index}: shape index.shape + [D]. */
    public static Tensor embedding(Tensor index, Tensor weight) {
        return Embedding.apply(index, weight, -1);
    }

    /** As {@link #embedding(Tensor, Tensor)}; row {@code paddingIdx} receives no gradient. */
    public static Tensor embedding(Tensor index, Tensor weight, int paddingIdx) {
        Tensor y = Embedding.apply(index, weight, paddingIdx);
        // FP16 activations: the rows are FP16 from here, so the residual stream that starts at an
        // embedding stays FP16 instead of upcasting every FP16 operand it is added to
        return Autocast.halfActivations() && y.dtype() == DType.F32 ? y.to(DType.F16) : y;
    }

    /**
     * softmax(scores * scale) over the last axis of attention scores [..., T, S], in one kernel;
     * {@code causal}: query t sees keys 0 .. t, as PyTorch's {@code is_causal}.
     */
    public static Tensor attentionSoftmax(Tensor scores, float scale, boolean causal) {
        return AttentionSoftmax.apply(scores, scale, causal);
    }

    /**
     * Quantises x to {@code bits} (8 or 16) bits and back:
     * clamp(round_half_even(x / scale), lowest, lowest + 2^bits - 1) * scale, {@code lowest} being
     * the lowest code minus the zero point (ONNX QuantizeLinear followed by DequantizeLinear).
     */
    public static Tensor fakeQuantize(Tensor x, float scale, float lowest, int bits) {
        ai.beestream.jitnn.ops.Unary.Kind kind = switch (bits) {
            case 8 -> ai.beestream.jitnn.ops.Unary.Kind.FAKE_QUANT8;
            case 16 -> ai.beestream.jitnn.ops.Unary.Kind.FAKE_QUANT16;
            default -> throw new IllegalArgumentException("fakeQuantize: 8 or 16 bits, not " + bits);
        };
        return ai.beestream.jitnn.ops.Unary.apply(kind, x, scale, lowest);
    }

    /**
     * Scaled dot-product attention, PyTorch semantics: softmax(q k^T * scale + mask) v, for q
     * [..., T, E], k [..., S, E], v [..., S, Ev]. {@code mask} is an additive mask broadcast to
     * [..., T, S] (0 to keep, a large negative number to drop), or null; {@code causal} lets
     * query t see keys 0 .. t (PyTorch's is_causal). A negative {@code scale} means 1 / sqrt(E).
     */
    public static Tensor scaledDotProductAttention(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        float s = scale < 0 ? (float) (1.0 / Math.sqrt(q.dim(-1))) : scale;
        if (q.rank() == 4 && k.rank() == 4 && k.dim(1) != q.dim(1)) {
            k = repeatHeads(k, q.dim(1), 1);
            v = repeatHeads(v, q.dim(1), 1);
        }
        if (FUSED_ATTENTION && FlashAttention.supports(q, k, v)) {
            return FlashAttention.apply(q, k, v, mask, causal, s);
        }
        Tensor scores = q.matmul(k, false, true);
        Tensor p;
        if (mask != null) {
            p = AttentionSoftmax.apply(scores.mul(s).add(mask), 1.0f, causal);
        } else {
            p = AttentionSoftmax.apply(scores, s, causal);
        }
        return p.matmul(v);
    }

    /**
     * Multi-head self-attention on x [B, T, D], as PyTorch's {@code nn.MultiheadAttention} with
     * batch_first: q, k, v = split(x Wqkv^T + bqkv), {@code heads} heads, then the output projection.
     * Wqkv is [3D, D] (rows q, then k, then v), Wout [D, D]; biases may be null.
     */
    public static Tensor multiHeadAttention(Tensor x, Tensor wqkv, Tensor bqkv, Tensor wout, Tensor bout, int heads, Tensor mask, boolean causal) {
        int b = x.dim(0);
        int t = x.dim(1);
        int dim = x.dim(2);
        int hd = dim / heads;
        if (hd * heads != dim) {
            throw new IllegalArgumentException(dim + " features do not split into " + heads + " heads");
        }
        if (FUSED_ATTENTION && hd <= FlashAttention.MAX_HEAD) {
            // the fused kernels read q, k and v straight from the packed projection and write [B, T, D]
            Tensor y = FlashAttention.packed(linear(x, wqkv, bqkv), heads, mask, causal, (float) (1.0 / Math.sqrt(hd)));
            return linear(y, wout, bout);
        }
        // [B, T, 3, H, hd] -> [3, B, H, T, hd]
        Tensor[] qkv = linear(x, wqkv, bqkv).reshape(b, t, 3, heads, hd).permute(2, 0, 3, 1, 4).chunk(3, 0);
        Tensor q = qkv[0].reshape(b, heads, t, hd);
        Tensor k = qkv[1].reshape(b, heads, t, hd);
        Tensor v = qkv[2].reshape(b, heads, t, hd);
        Tensor y = scaledDotProductAttention(q, k, v, mask, causal, -1.0f);
        return linear(y.permute(0, 2, 1, 3).reshape(b, t, dim), wout, bout);
    }

    /**
     * Scaled dot-product attention on q [B, T, H, E], k and v [B, S, H, E] (heads after tokens, as
     * the projections produce them), returning [B, T, H, E]; otherwise as
     * {@link #scaledDotProductAttention}. The fused kernels read this layout directly.
     */
    public static Tensor scaledDotProductAttentionBTHD(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        float s = scale < 0 ? (float) (1.0 / Math.sqrt(q.dim(-1))) : scale;
        if (q.rank() == 4 && k.rank() == 4 && k.dim(2) != q.dim(2)) {
            k = repeatHeads(k, q.dim(2), 2);
            v = repeatHeads(v, q.dim(2), 2);
        }
        if (FUSED_ATTENTION && (q.dtype() == DType.F32 || q.dtype() == DType.F16) && q.dim(-1) <= FlashAttention.MAX_HEAD && k.shape().equals(v.shape())) {
            return FlashAttention.bthd(q, k, v, mask, causal, s);
        }
        Tensor y = scaledDotProductAttention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), mask, causal, s);
        return y.permute(0, 2, 1, 3);
    }

    /**
     * Grouped-query attention: the key or value heads at {@code axis} repeated so each serves
     * {@code heads / x.dim(axis)} consecutive query heads (PyTorch's enable_gqa; query head h reads
     * key head h / group). The gradient sums over each group.
     */
    static Tensor repeatHeads(Tensor x, int heads, int axis) {
        int kv = x.dim(axis);
        if (heads % kv != 0) {
            throw new IllegalArgumentException("grouped-query attention: " + heads + " query heads over " + kv + " key/value heads");
        }
        int[] d = x.shape().dims();
        int[] split = new int[d.length + 1];
        int[] out = d.clone();
        for (int i = 0, j = 0; i < d.length; i++) {
            split[j++] = d[i];
            if (i == axis) {
                split[j++] = 1;
            }
        }
        int[] big = split.clone();
        big[axis + 1] = heads / kv;
        out[axis] = heads;
        return x.reshape(split).expand(Shape.of(big)).reshape(out);
    }

    /** Rotary position embedding of x [..., T, H, D] (the heads of a token share its position); see {@link #rope}. */
    public static Tensor ropeHeads(Tensor x, float base, int offset, boolean interleaved) {
        return Rope.applyHeads(x, base, offset, interleaved);
    }

    /** {@link #ropeHeads} with the frequencies rescaled for longer contexts (Llama 3.1's rope_scaling; null for none). */
    public static Tensor ropeHeads(Tensor x, float base, RopeScaling scaling, int offset, boolean interleaved) {
        return Rope.applyHeads(x, base, scaling, offset, interleaved);
    }

    /**
     * Rotary position embedding of x [..., T, D] at positions offset .. offset + T - 1, frequency
     * base^(-2i / D) for pair i. {@code interleaved} pairs (2i, 2i + 1) as GPT-J and Meta's Llama
     * code; otherwise (i, i + D/2), the rotate-half form of Hugging Face's Llama.
     */
    public static Tensor rope(Tensor x, float base, int offset, boolean interleaved) {
        return Rope.apply(x, base, offset, interleaved);
    }

    /**
     * One INT32 index per row of non-negative weights [..., V] (probabilities, or any unnormalised
     * weights), drawn in proportion to the weight. {@code seed} is an INT32 [2] host tensor; a new
     * seed draws new samples without recompiling.
     */
    public static Tensor multinomial(Tensor weights, Tensor seed, int stream) {
        return Sampling.multinomial(weights, seed, stream);
    }

    /** How a loss reduces its per-element values. */
    public enum Reduction {
        /** The per-element losses, unreduced. */
        NONE,
        /** The mean of the per-element losses. */
        MEAN,
        /** The sum of the per-element losses. */
        SUM
    }

    private static Tensor reduce(Tensor l, Reduction r) {
        return switch (r) {
            case NONE -> l;
            case MEAN -> l.mean();
            case SUM -> l.sum();
        };
    }

    /** Mean squared error between {@code input} and {@code target} (same shape). */
    public static Tensor mseLoss(Tensor input, Tensor target) {
        return mseLoss(input, target, Reduction.MEAN);
    }

    /**
     * Squared error between {@code input} and {@code target} (same shape), reduced as {@code reduction}
     * says.
     */
    public static Tensor mseLoss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(input.sub(target).square(), reduction);
    }

    /**
     * Binary cross-entropy of probabilities {@code p} in [0, 1] against {@code target}, PyTorch
     * semantics: the logs are clamped at -100, so p = 0 or 1 gives finite values and gradients.
     */
    public static Tensor binaryCrossEntropy(Tensor p, Tensor target) {
        return binaryCrossEntropy(p, target, Reduction.MEAN);
    }

    /** {@link #binaryCrossEntropy(Tensor, Tensor)} reduced as {@code reduction} says. */
    public static Tensor binaryCrossEntropy(Tensor p, Tensor target, Reduction reduction) {
        return reduce(Binary.apply(Binary.Kind.BCE, p, target), reduction);
    }

    /** Binary cross-entropy on logits, sigmoid folded in without overflow (PyTorch's BCEWithLogits). */
    public static Tensor binaryCrossEntropyWithLogits(Tensor logits, Tensor target) {
        return binaryCrossEntropyWithLogits(logits, target, Reduction.MEAN);
    }

    /** {@link #binaryCrossEntropyWithLogits(Tensor, Tensor)} reduced as {@code reduction} says. */
    public static Tensor binaryCrossEntropyWithLogits(Tensor logits, Tensor target, Reduction reduction) {
        return reduce(Binary.apply(Binary.Kind.BCE_LOGITS, logits, target), reduction);
    }

    /** Mean absolute error (L1), PyTorch's l1_loss. */
    public static Tensor l1Loss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(input.sub(target).abs(), reduction);
    }

    /**
     * Huber loss with threshold {@code delta}: 0.5 d^2 where |d| &lt; delta, else delta (|d| - delta / 2),
     * d = input - target (PyTorch's huber_loss).
     */
    public static Tensor huberLoss(Tensor input, Tensor target, float delta, Reduction reduction) {
        Tensor d = input.sub(target);
        Tensor a = d.abs();
        Tensor quadratic = d.square().mul(0.5f);
        Tensor linear = a.add(-0.5f * delta).mul(delta);
        return reduce(where(a.lt(Tensor.scalar(delta)), quadratic, linear), reduction);
    }

    /** Smooth L1 loss, PyTorch's smooth_l1_loss: Huber with delta = beta, divided by beta. */
    public static Tensor smoothL1Loss(Tensor input, Tensor target, float beta, Reduction reduction) {
        return huberLoss(input, target, beta, reduction).mul(1.0f / beta);
    }

    /** Negative log-likelihood of INT32 labels under log-probabilities [N, C], reduced. */
    public static Tensor nllLoss(Tensor logp, Tensor labels, Reduction reduction) {
        return reduce(nll(logp, labels), reduction);
    }

    /** Hinge loss with targets in {-1, 1}: max(0, 1 - target * input), Keras's hinge (mean). */
    public static Tensor hingeLoss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(target.mul(input).neg().add(1.0f).relu(), reduction);
    }

    /** Squared hinge loss: max(0, 1 - target * input)^2. */
    public static Tensor squaredHingeLoss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(target.mul(input).neg().add(1.0f).relu().square(), reduction);
    }

    /**
     * Cosine similarity along {@code axis}: a . b / (max(|a|, eps) max(|b|, eps)), PyTorch's
     * cosine_similarity.
     */
    public static Tensor cosineSimilarity(Tensor a, Tensor b, int axis, float eps) {
        Tensor eps0 = Tensor.scalar(eps);
        Tensor na = a.square().sum(axis, false).sqrt().maximum(eps0);
        Tensor nb = b.square().sum(axis, false).sqrt().maximum(eps0);
        return a.mul(b).sum(axis, false).div(na.mul(nb));
    }

    /** Cosine proximity loss (DL4J, Keras): minus the cosine similarity along the last axis, reduced. */
    public static Tensor cosineProximityLoss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(cosineSimilarity(input, target, -1, 1e-8f).neg(), reduction);
    }

    /**
     * Poisson negative log-likelihood, PyTorch's poisson_nll_loss without the Stirling term:
     * exp(input) - target input with {@code logInput}, else input - target log(input + eps).
     */
    public static Tensor poissonNllLoss(Tensor input, Tensor target, boolean logInput, float eps, Reduction reduction) {
        Tensor l = logInput ? input.exp().sub(target.mul(input)) : input.sub(target.mul(input.add(eps).log()));
        return reduce(l, reduction);
    }

    /**
     * CTC loss, PyTorch's ctc_loss: log-probabilities [T, B, C] (log_softmax over C), INT32 targets
     * [B, L] padded, input and target lengths; blank index {@code blank}. MEAN divides each sample's
     * loss by its target length, then averages (PyTorch's default).
     */
    public static Tensor ctcLoss(Tensor logProbs, Tensor targets, int[] inputLengths, int[] targetLengths, int blank, Reduction reduction) {
        int b = logProbs.dim(1);
        Tensor l = ai.beestream.jitnn.ops.Ctc.apply(logProbs, targets, Tensor.of(inputLengths, b), Tensor.of(targetLengths, b), blank);
        if (reduction == Reduction.MEAN) {
            float[] inv = new float[b];
            for (int i = 0; i < b; i++) {
                inv[i] = 1.0f / Math.max(targetLengths[i], 1);
            }
            return l.mul(Tensor.of(inv, b)).mean();
        }
        return reduce(l, reduction);
    }

    /**
     * torch.nonzero_static: INT32 coordinates [size, rank] of the first {@code size} non-zero
     * elements, the rest of the rows {@code fill}. Unlike {@link Tensor#nonzero()} it needs no host
     * read, so it stays inside a compiled plan.
     */
    public static Tensor nonzero(Tensor x, int size, int fill) {
        return ai.beestream.jitnn.ops.Compact.nonzero(x, size, fill);
    }

    /**
     * torch.unique (sorted): the distinct values of x, then (when asked for, else null) the INT32
     * index of each element's value among them, shaped like x, and the INT32 count of each value.
     */
    public static Tensor[] unique(Tensor x, boolean returnInverse, boolean returnCounts) {
        return ai.beestream.jitnn.ops.Compact.unique(x, returnInverse, returnCounts);
    }

    /** log sigmoid(x) = -softplus(-x), without overflow. */
    public static Tensor logSigmoid(Tensor x) {
        return x.neg().softplus().neg();
    }

    /**
     * Negative-sampling loss of word2vec (Mikolov et al., 2013): hidden [B, D] (the centre word, or
     * the averaged context), target [B, D] (the output vector of the true word) and negatives [B, K, D]
     * (output vectors of sampled words); mean over B of -log sigmoid(h . t) - sum_k log sigmoid(-h . n_k).
     */
    public static Tensor negativeSamplingLoss(Tensor hidden, Tensor target, Tensor negatives) {
        Tensor pos = logSigmoid(hidden.mul(target).sum(-1, false));
        Tensor neg = logSigmoid(negatives.mul(hidden.unsqueeze(1)).sum(-1, false).neg()).sum(-1, false);
        return pos.add(neg).neg().mean();
    }

    /**
     * GloVe's weighted least squares (Pennington et al., 2014) over a batch of co-occurrences: mean
     * of weight (wi . wj + bi + bj - logX)^2, with wi, wj [B, D] and the rest [B].
     */
    public static Tensor gloveLoss(Tensor wi, Tensor wj, Tensor bi, Tensor bj, Tensor logX, Tensor weight) {
        Tensor diff = wi.mul(wj).sum(-1, false).add(bi).add(bj).sub(logX);
        return weight.mul(diff.square()).mean();
    }

    /** Euclidean distance of each row pair (last axis), PyTorch's pairwise_distance with eps 1e-6. */
    public static Tensor pairwiseDistance(Tensor a, Tensor b) {
        return a.sub(b).add(1e-6f).square().sum(-1, false).sqrt();
    }

    /** Triplet margin loss: max(d(anchor, positive) - d(anchor, negative) + margin, 0). */
    public static Tensor tripletMarginLoss(Tensor anchor, Tensor positive, Tensor negative, float margin, Reduction reduction) {
        return reduce(pairwiseDistance(anchor, positive).sub(pairwiseDistance(anchor, negative)).add(margin).relu(), reduction);
    }

    /**
     * Contrastive loss (Hadsell et al., 2006) with {@code similar} 1 for similar pairs and 0 for
     * dissimilar ones: similar d^2 + (1 - similar) max(margin - d, 0)^2.
     */
    public static Tensor contrastiveLoss(Tensor a, Tensor b, Tensor similar, float margin, Reduction reduction) {
        Tensor d = pairwiseDistance(a, b);
        Tensor apart = d.neg().add(margin).relu().square();
        return reduce(similar.mul(d.square()).add(similar.neg().add(1.0f).mul(apart)), reduction);
    }

    /** Cosine embedding loss: 1 - cos(a, b) for target 1, max(0, cos(a, b) - margin) for -1. */
    public static Tensor cosineEmbeddingLoss(Tensor a, Tensor b, Tensor target, float margin, Reduction reduction) {
        Tensor cos = cosineSimilarity(a, b, -1, 1e-8f);
        return reduce(where(target.gt(Tensor.scalar(0.0f)), cos.neg().add(1.0f), cos.add(-margin).relu()), reduction);
    }

    /** Margin ranking loss: max(0, -target (x1 - x2) + margin), target 1 or -1. */
    public static Tensor marginRankingLoss(Tensor x1, Tensor x2, Tensor target, float margin, Reduction reduction) {
        return reduce(target.neg().mul(x1.sub(x2)).add(margin).relu(), reduction);
    }

    /**
     * Sparsemax (Martins and Astudillo, 2016) along the last axis: the Euclidean projection onto the
     * probability simplex, max(x - tau, 0) with tau set by the sorted support.
     */
    public static Tensor sparsemax(Tensor x) {
        int n = x.dim(-1);
        Tensor z = x.sort(-1, true)[0];
        Tensor cum = z.cumsum(-1);
        float[] k = new float[n];
        for (int i = 0; i < n; i++) {
            k[i] = i + 1;
        }
        Tensor ks = Tensor.of(k, n);
        Tensor support = z.mul(ks).add(1.0f).gt(cum);
        Tensor size = support.sum(-1, true);
        Tensor tau = support.mul(z).sum(-1, true).add(-1.0f).div(size);
        return x.sub(tau).relu();
    }

    /** Mean squared logarithmic error: (log1p(input) - log1p(target))^2 (Keras's MSLE). */
    public static Tensor msleLoss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(input.log1p().sub(target.log1p()).square(), reduction);
    }

    /** Mean absolute percentage error: 100 |target - input| / max(|target|, eps) (Keras's MAPE, eps 1e-7). */
    public static Tensor mapeLoss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(target.sub(input).abs().div(target.abs().maximum(Tensor.scalar(1e-7f))).mul(100.0f), reduction);
    }

    /**
     * KL divergence of N(mu, exp(logvar)) from N(0, 1), summed over every element:
     * -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), the VAE regulariser.
     */
    public static Tensor gaussianKl(Tensor mu, Tensor logvar) {
        return logvar.add(1.0f).sub(mu.square()).sub(logvar.exp()).sum().mul(-0.5f);
    }

    /**
     * One LSTM layer over x [T, B, I] (time first), PyTorch's equations and gate order (i, f, g, o):
     * wih [4H, I], whh [4H, H], biases [4H], h0 and c0 [B, H]. Returns {y [T, B, H], h_n, c_n}
     * ([B, H] each); c_n carries no gradient.
     */
    public static Tensor[] lstm(Tensor x, Tensor wih, Tensor whh, Tensor bih, Tensor bhh, Tensor h0, Tensor c0) {
        return lstm(x, wih, whh, bih, bhh, h0, c0, null);
    }

    /**
     * As {@link #lstm(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)} for padded batches:
     * INT32 {@code lengths} [B] (null: all T); outputs past a sequence's length are 0 and h_n, c_n
     * are each sequence's last valid state.
     */
    public static Tensor[] lstm(Tensor x, Tensor wih, Tensor whh, Tensor bih, Tensor bhh, Tensor h0, Tensor c0, int[] lengths) {
        int t = x.dim(0);
        int b = x.dim(1);
        int h = whh.dim(1);
        Tensor xg = linear(x.reshape(t * b, x.dim(2)), wih, bih).reshape(t, b, 4 * h);
        Tensor[] r = Recurrent.lstm(xg, whh, bhh, h0, c0, lengths == null ? null : Tensor.of(lengths, b));
        return new Tensor[] { r[0], lastSteps(r[0], lengths), r[1] };
    }

    /** y [T, B, H] at each sequence's last valid step: [B, H]. */
    private static Tensor lastSteps(Tensor y, int[] lengths) {
        int t = y.dim(0);
        int b = y.dim(1);
        int h = y.dim(2);
        if (lengths == null) {
            return y.narrow(0, t - 1, 1).reshape(b, h);
        }
        int[] idx = new int[b * h];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = lengths[i / h] - 1;
        }
        return y.gather(0, Tensor.of(idx, 1, b, h)).reshape(b, h);
    }

    /**
     * x [T, B, ...] with each sequence reversed within its length (padding left in place), as a
     * bidirectional layer reads it; lengths null reverses the whole time axis.
     */
    public static Tensor reverseSequences(Tensor x, int[] lengths) {
        int t = x.dim(0);
        int b = x.dim(1);
        int inner = x.numel() / (t * b);
        int[] idx = new int[x.numel()];
        for (int s = 0; s < t; s++) {
            for (int bi = 0; bi < b; bi++) {
                int len = lengths == null ? t : lengths[bi];
                int src = s < len ? len - 1 - s : s;
                for (int q = 0; q < inner; q++) {
                    idx[(s * b + bi) * inner + q] = src;
                }
            }
        }
        return x.gather(0, Tensor.of(idx, x.shape().dims()));
    }

    /**
     * One GRU layer over x [T, B, I] (time first), PyTorch's equations and gate order (r, z, n):
     * wih [3H, I], whh [3H, H], biases [3H], h0 [B, H]. Returns {y [T, B, H], h_n [B, H]}.
     */
    public static Tensor[] gru(Tensor x, Tensor wih, Tensor whh, Tensor bih, Tensor bhh, Tensor h0) {
        return gru(x, wih, whh, bih, bhh, h0, null);
    }

    /** As {@link #gru(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)} for padded batches of {@code lengths}. */
    public static Tensor[] gru(Tensor x, Tensor wih, Tensor whh, Tensor bih, Tensor bhh, Tensor h0, int[] lengths) {
        int t = x.dim(0);
        int b = x.dim(1);
        int h = whh.dim(1);
        Tensor xg = linear(x.reshape(t * b, x.dim(2)), wih, bih).reshape(t, b, 3 * h);
        Tensor y = Recurrent.gru(xg, whh, bhh, h0, lengths == null ? null : Tensor.of(lengths, b));
        return new Tensor[] { y, lastSteps(y, lengths) };
    }

    /**
     * PReLU: x where x > 0, else a x, with a learnable slope {@code weight} of one element or one
     * per channel (axis 1 of x, as PyTorch).
     */
    public static Tensor prelu(Tensor x, Tensor weight) {
        Tensor a = weight;
        if (weight.numel() > 1 && x.rank() > 1) {
            int[] dims = new int[x.rank()];
            java.util.Arrays.fill(dims, 1);
            dims[1] = weight.numel();
            a = weight.reshape(dims);
        }
        return Binary.apply(Binary.Kind.PRELU, x, a);
    }

    /**
     * RReLU (Xu et al., 2015): in training, a slope drawn uniformly from [lower, upper) for every
     * element, seeded by the INT32 [2] {@code seed} tensor (see {@link Tensor#seed}); in
     * evaluation the mean slope (lower + upper) / 2.
     */
    public static Tensor rrelu(Tensor x, float lower, float upper, boolean training, Tensor seed) {
        if (!training) {
            return x.leakyRelu((lower + upper) / 2);
        }
        return Binary.apply(Binary.Kind.PRELU, x, Tensor.uniform(seed, lower, upper, x.shape().dims()));
    }

    /** cond != 0 ? a : b element-wise, the three broadcast together. */
    public static Tensor where(Tensor cond, Tensor a, Tensor b) {
        return ai.beestream.jitnn.ops.Select.where(cond, a, b);
    }

    /** INT32 labels [...] to a one-hot [..., classes]. */
    public static Tensor oneHot(Tensor labels, int classes) {
        return ai.beestream.jitnn.ops.Select.oneHot(labels, classes);
    }

    /** How {@link #embeddingBag} and {@link #segmentReduce} combine the rows of a bag. */
    public enum BagMode {
        /** The sum of the rows. */
        SUM,
        /** The mean of the rows. */
        MEAN,
        /** The element-wise maximum of the rows. */
        MAX
    }

    /**
     * Sum, mean or max of the row segments of x [rows, D]: segment b covers rows offsets[b] ..
     * offsets[b + 1] - 1, the last to the end; an empty segment gives zeros. [segments, D].
     */
    public static Tensor segmentReduce(Tensor x, int[] offsets, BagMode mode) {
        return ai.beestream.jitnn.ops.Indexing.segmentReduce(x, offsets, mode.ordinal());
    }

    /**
     * EmbeddingBag (PyTorch's, 1-D input with offsets): the rows of {@code weight} picked by the INT32
     * {@code indices}, combined per bag by {@code mode}. [bags, D].
     */
    public static Tensor embeddingBag(Tensor indices, int[] offsets, Tensor weight, BagMode mode) {
        return segmentReduce(embedding(indices, weight), offsets, mode);
    }

    /** FP32 x (last axis a multiple of 32) quantised to Q8_0, GGUF's 8-bit block format. */
    public static Tensor quantizeQ8(Tensor x) {
        return ai.beestream.jitnn.ops.Quant.quantizeQ8(x);
    }

    /** A Q8_0 tensor back to FP32. */
    public static Tensor dequantize(Tensor q) {
        return ai.beestream.jitnn.ops.Quant.dequantize(q);
    }

    /** y = x W^T + b with Q8_0 weights W [out, in]; the gradient reaches x, not W. */
    public static Tensor linearQ8(Tensor x, Tensor weight, Tensor bias) {
        return ai.beestream.jitnn.ops.Quant.linear(x, weight, bias);
    }

    /** x reversed along each of {@code axes} (torch.flip). */
    public static Tensor flip(Tensor x, int... axes) {
        Tensor y = x;
        for (int axis : axes) {
            int n = x.dim(axis);
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) {
                idx[i] = n - 1 - i;
            }
            y = y.indexSelect(axis, Tensor.of(idx, n));
        }
        return y;
    }

    /** The central height x width window of images [..., H, W] (torchvision's center_crop, top-left rounding down). */
    public static Tensor centerCrop(Tensor x, int height, int width) {
        int h = x.dim(-2);
        int w = x.dim(-1);
        if (height > h || width > w) {
            throw new IllegalArgumentException("crop " + height + "x" + width + " larger than " + x.shape());
        }
        int top = (int) Math.round((h - height) / 2.0);
        int left = (int) Math.round((w - width) / 2.0);
        return x.narrow(-2, top, height).narrow(-1, left, width);
    }

    /** Per-channel (x - mean) / std of images [N, C, H, W] (torchvision's normalize). */
    /** x / max(||x||_p, eps) along {@code axis}, p = 1 or 2 (torch.nn.functional.normalize; ONNX LpNormalization with eps 0). */
    public static Tensor lpNormalize(Tensor x, int p, int axis, float eps) {
        Tensor norm = x.norm(p, axis, true);
        return x.div(eps > 0 ? norm.maximum(Tensor.scalar(eps)) : norm);
    }

    /**
     * Moves blocks of channels into space: x [N, C, H, W] to [N, C / b^2, H b, W b]. Mode "DCR"
     * (depth, column, row; ONNX's default) or "CRD" (column, row, depth: torch pixel_shuffle).
     */
    public static Tensor depthToSpace(Tensor x, int block, String mode) {
        if (x.rank() != 4) {
            throw new IllegalArgumentException("depthToSpace: needs [N, C, H, W], got " + x.shape());
        }
        int n = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3), b = block;
        if (c % (b * b) != 0) {
            throw new IllegalArgumentException("depthToSpace: " + c + " channels are not a multiple of " + b * b);
        }
        int co = c / (b * b);
        Tensor t = switch (mode) {
            case "DCR" -> x.reshape(n, b, b, co, h, w).permute(0, 3, 4, 1, 5, 2);
            case "CRD" -> x.reshape(n, co, b, b, h, w).permute(0, 1, 4, 2, 5, 3);
            default -> throw new IllegalArgumentException("depthToSpace: mode must be DCR or CRD, got " + mode);
        };
        return t.reshape(n, co, h * b, w * b);
    }

    /** Moves blocks of space into channels: x [N, C, H, W] to [N, C b^2, H / b, W / b] (ONNX SpaceToDepth; inverse of DCR depthToSpace). */
    public static Tensor spaceToDepth(Tensor x, int block) {
        if (x.rank() != 4) {
            throw new IllegalArgumentException("spaceToDepth: needs [N, C, H, W], got " + x.shape());
        }
        int n = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3), b = block;
        if (h % b != 0 || w % b != 0) {
            throw new IllegalArgumentException("spaceToDepth: " + h + "x" + w + " is not a multiple of " + b);
        }
        return x.reshape(n, c, h / b, b, w / b, b).permute(0, 3, 5, 1, 2, 4).reshape(n, c * b * b, h / b, w / b);
    }

    /** Slices of data picked by INT32 index rows indices [..., k] (ONNX GatherND, batch_dims 0); differentiable in data. */
    public static Tensor gatherND(Tensor data, Tensor indices) {
        return ai.beestream.jitnn.ops.NdIndexing.gatherND(data, indices);
    }

    /**
     * A copy of data with the update slices written ("none"), added ("add") or multiplied ("mul")
     * at the INT32 index rows indices [..., k] (ONNX ScatterND); differentiable for none and add.
     */
    public static Tensor scatterND(Tensor data, Tensor indices, Tensor updates, String reduction) {
        return ai.beestream.jitnn.ops.NdIndexing.scatterND(data, indices, updates, reduction);
    }

    /** Folds columns back into an image, summing overlaps (torch fold, ONNX Col2Im); see {@link ai.beestream.jitnn.ops.Spatial#fold}. */
    public static Tensor fold(Tensor cols, int[] outputSize, int[] kernel, int[] stride, int[] pad, int[] dil) {
        return ai.beestream.jitnn.ops.Spatial.fold(cols, outputSize, kernel, stride, pad, dil);
    }

    /** Normalises images x [N, C, H, W] per channel: (x - mean[c]) / std[c], as torchvision's Normalize. */
    public static Tensor normalizeImage(Tensor x, float[] mean, float[] std) {
        int c = x.dim(1);
        if (mean.length != c || std.length != c) {
            throw new IllegalArgumentException("normalize: " + c + " channels, " + mean.length + " means, " + std.length + " deviations");
        }
        float[] scale = new float[c];
        for (int i = 0; i < c; i++) {
            scale[i] = 1.0f / std[i];
        }
        return x.sub(Tensor.of(mean, 1, c, 1, 1)).mul(Tensor.of(scale, 1, c, 1, 1));
    }

    /** RGB images [N, 3, H, W] to luminance [N, 1, H, W] with torchvision's weights (0.2989, 0.587, 0.114). */
    public static Tensor rgbToGrayscale(Tensor x) {
        if (x.dim(1) != 3) {
            throw new IllegalArgumentException("rgbToGrayscale needs 3 channels, got " + x.shape());
        }
        return x.mul(Tensor.of(new float[] { 0.2989f, 0.587f, 0.114f }, 1, 3, 1, 1)).sum(1, true);
    }

    /** SwiGLU gating, silu(gate) * up, as in the Llama feed-forward block. */
    public static Tensor swiglu(Tensor gate, Tensor up) {
        return gate.silu().mul(up);
    }

    /** Fraction of rows whose argmax equals the label; runs now and returns a host value. */
    public static float accuracy(Tensor logits, Tensor labels) {
        int[] predicted = logits.argmax(-1).toIntArray();
        int[] truth = labels.toIntArray();
        int correct = 0;
        for (int i = 0; i < truth.length; i++) {
            correct += predicted[i] == truth[i] ? 1 : 0;
        }
        return (float) correct / truth.length;
    }
}
