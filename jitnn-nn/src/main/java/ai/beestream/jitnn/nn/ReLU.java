package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** ReLU: max(x, 0). */
public final class ReLU extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.relu();
    }
}
