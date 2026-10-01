package org.beehive.jitnn;

import org.beehive.jitnn.ops.CrossEntropy;
import org.beehive.jitnn.ops.Linear;
import org.beehive.jitnn.ops.Nll;

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
