package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** RMSNorm over the last axis, with weight 1 initially, like PyTorch's {@code nn.RMSNorm}. */
public final class RMSNorm extends Module {

    private final Tensor weight;
    private final float eps;

    public RMSNorm(int features, float eps) {
        this.weight = parameter(Tensor.ones(features));
        this.eps = eps;
    }

    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.rmsNorm(x, weight, eps);
    }
}
