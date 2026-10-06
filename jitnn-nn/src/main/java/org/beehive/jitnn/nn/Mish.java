package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** Mish: x tanh(softplus(x)). */
public final class Mish extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.mish();
    }
}
