package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** HardSwish: x hardSigmoid(x). */
public final class HardSwish extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.hardSwish();
    }
}
