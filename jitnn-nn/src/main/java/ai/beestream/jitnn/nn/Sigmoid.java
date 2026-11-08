package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** The logistic sigmoid 1 / (1 + exp(-x)). */
public final class Sigmoid extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.sigmoid();
    }
}
