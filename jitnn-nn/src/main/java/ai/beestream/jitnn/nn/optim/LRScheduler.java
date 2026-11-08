package ai.beestream.jitnn.nn.optim;

import ai.beestream.jitnn.io.StateDict;

/**
 * Sets an optimiser's learning rate per epoch; call {@link #step()} once at the end of each
 * epoch, as with PyTorch's schedulers. The new rate takes effect without recompiling. The
 * warm-up schedules count optimiser steps instead: call {@link #step()} after every
 * {@code optimizer.step()}.
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

    /** The epoch (or step) count, for checkpoints; the rate itself is restored with the optimiser. */
    public StateDict stateDict() {
        return new StateDict().put("epoch", epoch);
    }

    public void loadStateDict(StateDict d) {
        epoch = d.intValue("epoch");
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

    /**
     * Linear warm-up from 0 to the base rate over {@code warmup} steps, then linear decay to 0 at
     * {@code total} steps (Hugging Face's get_linear_schedule_with_warmup). Sets the step-0 rate now.
     */
    public static LRScheduler warmupLinear(Optimizer optimizer, int warmup, int total) {
        return start(new LRScheduler(optimizer) {
            @Override
            protected float rate(int s) {
                if (s < warmup) {
                    return baseLr * s / Math.max(1, warmup);
                }
                return baseLr * Math.max(0.0f, (float) (total - s) / Math.max(1, total - warmup));
            }
        });
    }

    /**
     * Linear warm-up over {@code warmup} steps, then a half-cosine decay to {@code minRatio} times
     * the base rate at {@code total} steps (Hugging Face's get_cosine_schedule_with_warmup when
     * minRatio is 0). Sets the step-0 rate now.
     */
    public static LRScheduler warmupCosine(Optimizer optimizer, int warmup, int total, float minRatio) {
        return start(new LRScheduler(optimizer) {
            @Override
            protected float rate(int s) {
                if (s < warmup) {
                    return baseLr * s / Math.max(1, warmup);
                }
                double progress = Math.min(1.0, (double) (s - warmup) / Math.max(1, total - warmup));
                double cosine = 0.5 * (1 + Math.cos(Math.PI * progress));
                return (float) (baseLr * (minRatio + (1 - minRatio) * cosine));
            }
        });
    }

    private static LRScheduler start(LRScheduler s) {
        s.optimizer.setLearningRate(s.rate(0));
        return s;
    }
}
