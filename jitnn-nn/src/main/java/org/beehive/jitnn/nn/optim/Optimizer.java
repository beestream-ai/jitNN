package org.beehive.jitnn.nn.optim;

import java.util.List;

import org.beehive.jitnn.Tensor;

/** Updates parameters in place from their gradients. */
public abstract class Optimizer {

    protected final List<Tensor> params;
    protected float lr;

    protected Optimizer(List<Tensor> params, float lr) {
        this.params = List.copyOf(params);
        this.lr = lr;
        for (Tensor p : this.params) {
            if (!p.storage().isPersistent()) {
                throw new IllegalArgumentException("not a parameter: " + p);
            }
        }
    }

    /** Records one update of every parameter that has a gradient. */
    public abstract void step();

    public void zeroGrad() {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }

    public float learningRate() {
        return lr;
    }

    /** Changes the learning rate; takes effect at the next step without recompiling. */
    public void setLearningRate(float lr) {
        this.lr = lr;
    }
}
