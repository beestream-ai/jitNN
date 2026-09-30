package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

public final class ReLU extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.relu();
    }
}
