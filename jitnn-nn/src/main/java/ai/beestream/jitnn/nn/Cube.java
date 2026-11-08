package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** x^3. */
public final class Cube extends Module {


    @Override
    public Tensor forward(Tensor x) {
        return x.cube();
    }
}
