package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** Softsign: x / (1 + |x|). */
public final class Softsign extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.softsign();
    }
}
