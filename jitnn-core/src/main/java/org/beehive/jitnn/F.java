package org.beehive.jitnn;

import org.beehive.jitnn.ops.BatchNorm2d;
import org.beehive.jitnn.ops.Conv2d;
import org.beehive.jitnn.ops.CrossEntropy;
import org.beehive.jitnn.ops.Dropout;
import org.beehive.jitnn.ops.Embedding;
import org.beehive.jitnn.ops.Linear;
import org.beehive.jitnn.ops.Nll;
import org.beehive.jitnn.ops.Pool2d;

/** Functional operators that are not methods of {@link Tensor}. */
public final class F {

    private F() {
    }

    /** y = x W^T + b, with W stored [out, in] and b of shape [out] (b may be null); one fused GEMM. */
    public static Tensor linear(Tensor x, Tensor weight, Tensor bias) {
        return Linear.apply(x, weight, bias, false);
    }

    /** relu(x W^T + b), with the bias and ReLU fused into the GEMM epilogue. */
    public static Tensor linearRelu(Tensor x, Tensor weight, Tensor bias) {
        return Linear.apply(x, weight, bias, true);
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

    /** Rows of {@code weight} [V, D] picked by the INT32 {@code index}: shape index.shape + [D]. */
    public static Tensor embedding(Tensor index, Tensor weight) {
        return Embedding.apply(index, weight, -1);
    }

    /** As {@link #embedding(Tensor, Tensor)}; row {@code paddingIdx} receives no gradient. */
    public static Tensor embedding(Tensor index, Tensor weight, int paddingIdx) {
        return Embedding.apply(index, weight, paddingIdx);
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
