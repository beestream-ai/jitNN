package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** ELU with slope {@code alpha} for negative inputs (PyTorch's default 1). */
public final class ELU extends Module {

    private final float alpha;

    public ELU(float alpha) {
        this.alpha = alpha;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.elu(alpha);
    }
}
