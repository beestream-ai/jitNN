package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

public final class Sigmoid extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.sigmoid();
    }
}
