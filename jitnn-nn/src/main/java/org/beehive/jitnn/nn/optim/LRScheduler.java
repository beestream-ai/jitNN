package org.beehive.jitnn.nn.optim;

/**
 * Sets an optimiser's learning rate per epoch; call {@link #step()} once at the end of each
 * epoch, as with PyTorch's schedulers. The new rate takes effect without recompiling.
 */
public abstract class LRScheduler {

    protected final Optimizer optimizer;
    protected final float baseLr;
    protected int epoch;

    protected LRScheduler(Optimizer optimizer) {
        this.optimizer = optimizer;
        this.baseLr = optimizer.learningRate();
    }

    /** The learning rate for {@code epoch} (0-based). */
    protected abstract float rate(int epoch);

    public void step() {
        epoch++;
        optimizer.setLearningRate(rate(epoch));
    }

    public float currentRate() {
        return optimizer.learningRate();
    }

    /** lr = base * gamma^(epoch / stepSize). */
    public static LRScheduler step(Optimizer optimizer, int stepSize, float gamma) {
        return new LRScheduler(optimizer) {
            @Override
            protected float rate(int e) {
                return (float) (baseLr * Math.pow(gamma, e / stepSize));
            }
        };
    }

    /** lr = base * gamma^m where m counts the milestones passed. */
    public static LRScheduler multiStep(Optimizer optimizer, int[] milestones, float gamma) {
        int[] ms = milestones.clone();
        return new LRScheduler(optimizer) {
            @Override
            protected float rate(int e) {
                int passed = 0;
                for (int m : ms) {
                    passed += e >= m ? 1 : 0;
                }
                return (float) (baseLr * Math.pow(gamma, passed));
            }
        };
    }

    /** Cosine annealing from base to {@code minLr} over {@code tMax} epochs (PyTorch's closed form). */
    public static LRScheduler cosine(Optimizer optimizer, int tMax, float minLr) {
        return new LRScheduler(optimizer) {
            @Override
            protected float rate(int e) {
                return (float) (minLr + (baseLr - minLr) * (1 + Math.cos(Math.PI * Math.min(e, tMax) / tMax)) / 2);
            }
        };
    }
}
