package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** LayerNorm over the last axis, with weight 1 and bias 0 initially, like PyTorch's {@code nn.LayerNorm}. */
public final class LayerNorm extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final float eps;

    public LayerNorm(int features) {
        this(features, 1e-5f, true);
    }

    public LayerNorm(int features, float eps, boolean bias) {
        this.weight = parameter(Tensor.ones(features));
        this.bias = bias ? parameter(Tensor.zeros(features)) : null;
        this.eps = eps;
    }

    public Tensor weight() {
        return weight;
    }

    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.layerNorm(x, weight, bias, eps);
    }
}
