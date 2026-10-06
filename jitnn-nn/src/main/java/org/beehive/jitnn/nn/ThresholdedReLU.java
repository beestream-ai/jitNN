package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** Keras's ThresholdedReLU: x where x > theta, else 0. */
public final class ThresholdedReLU extends Module {

    private final float theta;

    public ThresholdedReLU(float theta) {
        this.theta = theta;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.thresholdedRelu(theta);
    }
}
