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

    /** Applies the layers; a Linear directly followed by a ReLU runs as one fused GEMM. */
    @Override
    public Tensor forward(Tensor x) {
        Tensor y = x;
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof Linear lin && i + 1 < layers.length && layers[i + 1] instanceof ReLU) {
                y = lin.forwardRelu(y);
                i++;
            } else {
                y = layers[i].forward(y);
            }
        }
        return y;
    }
}
