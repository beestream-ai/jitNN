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

    /** A one-element device flag that skips the next updates when non-zero (set by {@link GradScaler}), or null. */
    protected Tensor gate;

    /** Records one update of every parameter that has a gradient. */
    public abstract void step();

    void setGate(Tensor gate) {
        this.gate = gate;
    }

    /** Undoes the step counting of the last {@link #step()}, which a loss scaler skipped on the device. */
    void rewindStep() {
    }

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
