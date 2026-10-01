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

    /**
     * Applies the layers, fusing a ReLU into the layer before it: Linear + ReLU and Conv2d + ReLU
     * run as one GEMM with the activation in the epilogue, BatchNorm2d + ReLU as one kernel.
     */
    @Override
    public Tensor forward(Tensor x) {
        Tensor y = x;
        for (int i = 0; i < layers.length; i++) {
            boolean reluNext = i + 1 < layers.length && layers[i + 1] instanceof ReLU;
            if (reluNext && layers[i] instanceof Linear lin) {
                y = lin.forwardRelu(y);
                i++;
            } else if (reluNext && layers[i] instanceof Conv2d conv) {
                y = conv.forwardRelu(y);
                i++;
            } else if (reluNext && layers[i] instanceof BatchNorm2d bn) {
                y = bn.forwardRelu(y);
                i++;
            } else {
                y = layers[i].forward(y);
            }
        }
        return y;
    }
}
