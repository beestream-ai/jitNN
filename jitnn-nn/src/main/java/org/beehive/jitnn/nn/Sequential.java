package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** Applies its modules in order. */
public final class Sequential extends Module {

    private final Module[] layers;

    public Sequential(Module... layers) {
        this.layers = layers.clone();
        for (Module m : this.layers) {
            child(m);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor y = x;
        for (Module m : layers) {
            y = m.forward(y);
        }
        return y;
    }
}
