package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** ReLU6: min(max(x, 0), 6). */
public final class ReLU6 extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return x.relu6();
    }
}
