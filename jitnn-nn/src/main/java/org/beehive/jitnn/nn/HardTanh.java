package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** HardTanh: x clamped to [min, max]. */
public final class HardTanh extends Module {

    private final float min;
    private final float max;

    public HardTanh(float min, float max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.hardTanh(min, max);
    }
}
