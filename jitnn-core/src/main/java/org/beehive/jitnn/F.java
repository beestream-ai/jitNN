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
        return org.beehive.jitnn.ops.Lrn.apply(x, size, alpha, beta, k);
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

    /** Mean absolute error (L1), PyTorch's l1_loss. */
    public static Tensor l1Loss(Tensor input, Tensor target, Reduction reduction) {
        return reduce(input.sub(target).abs(), reduction);
    }

    /**
     * Huber loss with threshold {@code delta}: 0.5 d^2 where |d| < delta, else delta (|d| - delta / 2),
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
        return org.beehive.jitnn.ops.Select.where(cond, a, b);
    }

    /** INT32 labels [...] to a one-hot [..., classes]. */
    public static Tensor oneHot(Tensor labels, int classes) {
        return org.beehive.jitnn.ops.Select.oneHot(labels, classes);
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
