package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** CELU with {@code alpha} (PyTorch's default 1). */
public final class CELU extends Module {

    private final float alpha;

    public CELU(float alpha) {
        this.alpha = alpha;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.celu(alpha);
    }
}
