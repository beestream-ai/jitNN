package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * y = x W^T + b with W of shape [out, in]. Initialised like PyTorch's {@code nn.Linear}: weight
 * and bias uniform in [-1/sqrt(in), 1/sqrt(in)).
 */
public final class Linear extends Module {

    private final Tensor weight;
    private final Tensor bias;

    public Linear(int in, int out, long seed) {
        this(in, out, true, seed);
    }

    public Linear(int in, int out, boolean bias, long seed) {
        float bound = (float) (1.0 / Math.sqrt(in));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, out, in));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    public Tensor weight() {
        return weight;
    }

    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.linear(x, weight, bias);
    }
}
