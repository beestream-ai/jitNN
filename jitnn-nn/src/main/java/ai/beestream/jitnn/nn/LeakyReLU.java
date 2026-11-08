package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** LeakyReLU with a fixed negative slope (PyTorch's default 0.01). */
public final class LeakyReLU extends Module {

    private final float slope;

    /** LeakyReLU with slope 0.01. */
    public LeakyReLU() {
        this(0.01f);
    }

    /** LeakyReLU with the given negative slope. */
    public LeakyReLU(float slope) {
        this.slope = slope;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.leakyRelu(slope);
    }
}
