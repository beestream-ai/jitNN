package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** Keras's ThresholdedReLU: x where x > theta, else 0. */
public final class ThresholdedReLU extends Module {

    private final float theta;

    /** ThresholdedReLU with threshold {@code theta}. */
    public ThresholdedReLU(float theta) {
        this.theta = theta;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.thresholdedRelu(theta);
    }
}
