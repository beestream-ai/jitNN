package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** x^3. */
public final class Cube extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.cube();
    }
}
