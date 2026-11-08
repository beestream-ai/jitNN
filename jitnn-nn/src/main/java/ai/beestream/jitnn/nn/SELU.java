package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** SELU with PyTorch's constants. */
public final class SELU extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.selu();
    }
}
