package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

public final class Tanh extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.tanh();
    }
}
