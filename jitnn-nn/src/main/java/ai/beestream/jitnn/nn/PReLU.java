package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** PReLU with a learnable slope, one shared or one per channel, initialised to {@code init} (PyTorch: 0.25). */
public final class PReLU extends Module {

    private final Tensor weight;

    public PReLU(int parameters, float init) {
        this.weight = parameter(Tensor.full(init, parameters));
    }

    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.prelu(x, weight);
    }
}
