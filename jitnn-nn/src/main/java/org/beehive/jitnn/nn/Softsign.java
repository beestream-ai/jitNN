package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** Softsign: x / (1 + |x|). */
public final class Softsign extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.softsign();
    }
}
