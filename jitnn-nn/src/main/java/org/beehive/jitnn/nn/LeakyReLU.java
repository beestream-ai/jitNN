package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** LeakyReLU with a fixed negative slope (PyTorch's default 0.01). */
public final class LeakyReLU extends Module {

    private final float slope;

    public LeakyReLU() {
        this(0.01f);
    }

    public LeakyReLU(float slope) {
        this.slope = slope;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.leakyRelu(slope);
    }
}
