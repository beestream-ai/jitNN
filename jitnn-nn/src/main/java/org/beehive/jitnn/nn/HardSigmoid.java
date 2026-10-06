package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** HardSigmoid: clamp(x / 6 + 1/2, 0, 1). */
public final class HardSigmoid extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.hardSigmoid();
    }
}
