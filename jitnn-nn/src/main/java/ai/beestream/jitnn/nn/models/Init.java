package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.Tensor;

/** Initialisers that another framework can reproduce bit for bit (Philox uniforms, scripts/pytorch/jitnn_init.py). */
final class Init {

    private Init() {
    }

    /** Uniform with mean 0 and standard deviation {@code std}: U(-std * sqrt(3), std * sqrt(3)). */
    static Tensor uniformStd(long seed, float std, int... dims) {
        float a = (float) (std * Math.sqrt(3.0));
        return Tensor.uniform(seed, -a, a, dims);
    }
}
