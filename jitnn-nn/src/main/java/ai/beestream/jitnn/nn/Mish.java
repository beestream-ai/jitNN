package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** Mish: x tanh(softplus(x)). */
public final class Mish extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.mish();
    }
}
