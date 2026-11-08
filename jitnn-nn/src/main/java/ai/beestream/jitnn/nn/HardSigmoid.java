package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** HardSigmoid: clamp(x / 6 + 1/2, 0, 1). */
public final class HardSigmoid extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.hardSigmoid();
    }
}
