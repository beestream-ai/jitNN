package org.beehive.jitnn.nn.optim;

import java.util.List;

import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;

/**
 * Dynamic loss scaling for FP16 training, as PyTorch's {@code torch.amp.GradScaler}: the loss is
 * multiplied by a scale before backward; {@link #unscale} divides the gradients and finds
 * overflows on the device; {@link #step} runs the optimiser, which skips the update on the
 * device when an overflow was found; {@link #update} (after the step has run, e.g. after
 * {@code loss.item()}) halves the scale on an overflow, or doubles it after
 * {@code growthInterval} clean steps. The scale reaches the device as a fresh host input each
 * step, so changing it never recompiles.
 *
 * <pre>{@code
 * Tensor loss = ...;
 * scaler.scale(loss).backward();
 * scaler.unscale(opt);              // optional: before clipping
 * Grad.clipNorm(params, 1.0f);
 * scaler.step(opt);
 * float value = loss.item();
 * scaler.update();
 * }</pre>
 */
public final class GradScaler {

    private float scale;
    private final float growthFactor;
    private final float backoffFactor;
    private final int growthInterval;
    private int cleanSteps;
    private Tensor found;
    private boolean unscaled;
    private Optimizer last;

    public GradScaler() {
        this(65536.0f, 2.0f, 0.5f, 2000);
    }

    public GradScaler(float initScale, float growthFactor, float backoffFactor, int growthInterval) {
        this.scale = initScale;
        this.growthFactor = growthFactor;
        this.backoffFactor = backoffFactor;
        this.growthInterval = growthInterval;
    }

    public float scale() {
        return scale;
    }

    /** loss * scale. */
    public Tensor scale(Tensor loss) {
        return loss.mul(Tensor.of(new float[] { scale }, 1));
    }

    /** Divides the gradients of {@code opt}'s parameters by the scale and records whether any overflowed. */
    public void unscale(Optimizer opt) {
        if (unscaled) {
            throw new IllegalStateException("unscale was already called for this step");
        }
        found = Grad.unscale(opt.params, Tensor.of(new float[] { 1.0f / scale }, 1));
        unscaled = true;
    }

    /** Unscales (unless done already) and records the optimiser step, skipped on the device on an overflow. */
    public void step(Optimizer opt) {
        if (!unscaled) {
            unscale(opt);
        }
        opt.setGate(found);
        try {
            opt.step();
        } finally {
            opt.setGate(null);
        }
        last = opt;
    }

    /**
     * Adjusts the scale from this step's overflow flag (reading it waits for the step) and
     * returns whether the step's update was skipped.
     */
    public boolean update() {
        if (found == null) {
            throw new IllegalStateException("update() needs a step() first");
        }
        boolean overflow = found.item() != 0.0f;
        if (overflow) {
            scale *= backoffFactor;
            cleanSteps = 0;
            last.rewindStep();
        } else if (++cleanSteps == growthInterval) {
            scale *= growthFactor;
            cleanSteps = 0;
        }
        found = null;
        unscaled = false;
        return overflow;
    }
}
