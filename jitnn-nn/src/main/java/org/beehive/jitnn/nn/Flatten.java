package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** [N, d1, d2, ...] to [N, d1 * d2 * ...], a view. */
public final class Flatten extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.reshape(x.dim(0), -1);
    }
}
