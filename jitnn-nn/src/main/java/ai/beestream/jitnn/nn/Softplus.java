package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** Softplus: log(1 + exp(x)). */
public final class Softplus extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.softplus();
    }
}
