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

    /** Gradient statistics [norm, factor, skip] for the next step (see GradStats), or null. */
    private Tensor stats;
    /** A one-element factor applied to every gradient at the next step (a loss scaler's 1 / scale), or null. */
    private Tensor preScale;

    /** Records one update of every parameter that has a gradient. */
    public abstract void step();

    /**
     * Clips the gradients of this optimiser's parameters to a global L2 norm of at most
     * {@code maxNorm} at the next {@link #step()}, as PyTorch's {@code clip_grad_norm_}, and returns
     * the norm before clipping (after loss unscaling). One small reduction per gradient; the
     * gradients are scaled inside the update itself. Reading the norm is a sync point.
     */
    public Tensor clipGradNorm(float maxNorm) {
        stats = computeStats(maxNorm, true);
        return stats.narrow(0, 0, 1).reshape();
    }

    private Tensor computeStats(float maxNorm, boolean clip) {
        List<Tensor> grads = new java.util.ArrayList<>();
        for (Tensor p : params) {
            if (p.grad() != null) {
                grads.add(p.grad());
            }
        }
        return org.beehive.jitnn.ops.GradStats.apply(grads, preScale, maxNorm, clip);
    }

    void setPreScale(Tensor factor) {
        this.preScale = factor;
    }

    /** The statistics the next update uses (computed now when a pre-scale is pending), or null; clears them. */
    protected Tensor takeStats() {
        Tensor s = stats;
        if (s == null && preScale != null) {
            s = computeStats(0.0f, false);
        }
        stats = null;
        preScale = null;
        lastStats = s;
        return s;
    }

    private Tensor lastStats;

    /** The statistics used by the last {@link #step()}, or null. */
    Tensor lastStats() {
        return lastStats;
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
