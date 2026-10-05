package org.beehive.jitnn;

import org.beehive.jitnn.ops.AttentionSoftmax;
import org.beehive.jitnn.ops.BatchNorm2d;
import org.beehive.jitnn.ops.Binary;
import org.beehive.jitnn.ops.Conv2d;
import org.beehive.jitnn.ops.ConvTranspose2d;
import org.beehive.jitnn.ops.CrossEntropy;
import org.beehive.jitnn.ops.Dropout;
import org.beehive.jitnn.ops.Embedding;
import org.beehive.jitnn.ops.FlashAttention;
import org.beehive.jitnn.ops.Linear;
import org.beehive.jitnn.ops.Nll;
import org.beehive.jitnn.ops.Pool2d;
import org.beehive.jitnn.ops.Recurrent;
import org.beehive.jitnn.ops.Rope;
import org.beehive.jitnn.ops.Sampling;
import org.beehive.jitnn.ops.RowNorm;
import org.beehive.jitnn.ops.Upsample2d;
import org.beehive.jitnn.kernels.ResizeKernels;

/** Functional operators that are not methods of {@link Tensor}. */
public final class F {

    /** Fused attention kernels for FP32 attention (default); {@code -Djitnn.fusedAttention=false} composes GEMMs and softmax instead. */
    private static final boolean FUSED_ATTENTION = !"false".equals(System.getProperty("jitnn.fusedAttention"));

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
        x = Autocast.operand(x);
        weight = Autocast.operand(weight);
        if (x.rank() <= 2) {
            return Linear.apply(x, weight, bias, relu);
        }
        int[] dims = x.shape().dims();
        Tensor y = Linear.apply(x.reshape(-1, dims[dims.length - 1]), weight, bias, relu);
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
        return Conv2d.apply(x, weight, bias, stride, padding, dilation, groups, false);
    }

    /** relu(conv2d(...)) with the bias and ReLU fused into the convolution's output. */
    public static Tensor conv2dRelu(Tensor x, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return Conv2d.apply(x, weight, bias, stride, padding, dilation, groups, true);
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

    public static Tensor maxPool2d(Tensor x, int kernel, int stride, int padding) {
        return Pool2d.max(x, kernel, stride, padding);
    }

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
        return Embedding.apply(index, weight, paddingIdx);
    }

    /**
     * Scaled dot-product attention, PyTorch semantics: softmax(q k^T * scale + mask) v, for q
     * [..., T, E], k [..., S, E], v [..., S, Ev]. {@code mask} is an additive mask broadcast to
     * [..., T, S] (0 to keep, a large negative number to drop), or null; {@code causal} lets
     * query t see keys 0 .. t (PyTorch's is_causal). A negative {@code scale} means 1 / sqrt(E).
     */
    public static Tensor scaledDotProductAttention(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        float s = scale < 0 ? (float) (1.0 / Math.sqrt(q.dim(-1))) : scale;
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
        if (FUSED_ATTENTION && q.dtype() == DType.F32 && q.dim(-1) <= FlashAttention.MAX_HEAD && k.shape().equals(v.shape())) {
            return FlashAttention.bthd(q, k, v, mask, causal, s);
        }
        Tensor y = scaledDotProductAttention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), mask, causal, s);
        return y.permute(0, 2, 1, 3);
    }

    /** Rotary position embedding of x [..., T, H, D] (the heads of a token share its position); see {@link #rope}. */
    public static Tensor ropeHeads(Tensor x, float base, int offset, boolean interleaved) {
        return Rope.applyHeads(x, base, offset, interleaved);
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
        NONE, MEAN, SUM
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

    public static Tensor binaryCrossEntropy(Tensor p, Tensor target, Reduction reduction) {
        return reduce(Binary.apply(Binary.Kind.BCE, p, target), reduction);
    }

    /** Binary cross-entropy on logits, sigmoid folded in without overflow (PyTorch's BCEWithLogits). */
    public static Tensor binaryCrossEntropyWithLogits(Tensor logits, Tensor target) {
        return binaryCrossEntropyWithLogits(logits, target, Reduction.MEAN);
    }

    public static Tensor binaryCrossEntropyWithLogits(Tensor logits, Tensor target, Reduction reduction) {
        return reduce(Binary.apply(Binary.Kind.BCE_LOGITS, logits, target), reduction);
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
        int t = x.dim(0);
        int b = x.dim(1);
        int h = whh.dim(1);
        Tensor xg = linear(x.reshape(t * b, x.dim(2)), wih, bih).reshape(t, b, 4 * h);
        Tensor[] r = Recurrent.lstm(xg, whh, bhh, h0, c0);
        return new Tensor[] { r[0], r[0].narrow(0, t - 1, 1).reshape(b, h), r[1] };
    }

    /**
     * One GRU layer over x [T, B, I] (time first), PyTorch's equations and gate order (r, z, n):
     * wih [3H, I], whh [3H, H], biases [3H], h0 [B, H]. Returns {y [T, B, H], h_n [B, H]}.
     */
    public static Tensor[] gru(Tensor x, Tensor wih, Tensor whh, Tensor bih, Tensor bhh, Tensor h0) {
        int t = x.dim(0);
        int b = x.dim(1);
        int h = whh.dim(1);
        Tensor xg = linear(x.reshape(t * b, x.dim(2)), wih, bih).reshape(t, b, 3 * h);
        Tensor y = Recurrent.gru(xg, whh, bhh, h0);
        return new Tensor[] { y, y.narrow(0, t - 1, 1).reshape(b, h) };
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
