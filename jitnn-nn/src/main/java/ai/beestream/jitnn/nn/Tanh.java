package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

public final class Tanh extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.tanh();
    }
}
