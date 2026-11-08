package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * y = x W^T + b with W of shape [out, in]. Initialised like PyTorch's {@code nn.Linear}: weight
 * and bias uniform in [-1/sqrt(in), 1/sqrt(in)).
 */
public final class Linear extends Module {

    private final Tensor weight;
    private final Tensor bias;

    /** A linear layer from {@code in} to {@code out} features with a bias, initialised from {@code seed}. */
    public Linear(int in, int out, long seed) {
        this(in, out, true, seed);
    }

    /**
     * A linear layer from {@code in} to {@code out} features, with or without a bias, initialised from
     * {@code seed}.
     */
    public Linear(int in, int out, boolean bias, long seed) {
        float bound = (float) (1.0 / Math.sqrt(in));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, out, in));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    /** The weight [out, in]. */
    public Tensor weight() {
        return weight;
    }

    /** The bias [out], or null without one. */
    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.linear(x, weight, bias);
    }

    /** relu(forward(x)) as one fused GEMM; {@link Sequential} uses it for a Linear followed by ReLU. */
    public Tensor forwardRelu(Tensor x) {
        return F.linearRelu(x, weight, bias);
    }
}
