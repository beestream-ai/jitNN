package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** Softplus: log(1 + exp(x)). */
public final class Softplus extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.softplus();
    }
}
