package ai.beestream.jitnn.nn.optim;

import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.StateDict;

/**
 * Dynamic loss scaling for FP16 training, as PyTorch's {@code torch.amp.GradScaler}: the loss is
 * multiplied by a scale before backward; {@link #unscale} makes the optimiser divide the
 * gradients by it (inside the update, together with any clipping) and look for overflows on the
 * device; {@link #step} runs the optimiser, which skips the update on the device when an overflow
 * was found; {@link #update} (after the step has run, e.g. after {@code loss.item()}) halves the
 * scale on an overflow, or doubles it after {@code growthInterval} clean steps. The scale reaches
 * the device as a fresh host input each step, so changing it never recompiles.
 *
 * <pre>{@code
 * Tensor loss = ...;
 * scaler.scale(loss).backward();
 * scaler.unscale(opt);              // optional: before clipping
 * opt.clipGradNorm(1.0f);
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
    private boolean unscaled;
    private Optimizer last;

    /**
     * A scaler with PyTorch's defaults: initial scale 65536, growth 2 after 2000 clean steps, backoff 0.5.
     */
    public GradScaler() {
        this(65536.0f, 2.0f, 0.5f, 2000);
    }

    /**
     * A scaler starting at {@code initScale} that multiplies the scale by {@code growthFactor} after
     * {@code growthInterval} steps without overflow and by {@code backoffFactor} after an overflow.
     */
    public GradScaler(float initScale, float growthFactor, float backoffFactor, int growthInterval) {
        this.scale = initScale;
        this.growthFactor = growthFactor;
        this.backoffFactor = backoffFactor;
        this.growthInterval = growthInterval;
    }

    /** The current loss scale. */
    public float scale() {
        return scale;
    }

    /** loss * scale. */
    public Tensor scale(Tensor loss) {
        return loss.mul(Tensor.of(new float[] { scale }, 1));
    }

    /** Makes {@code opt} divide its gradients by the scale at the next step (and clip after unscaling). */
    public void unscale(Optimizer opt) {
        if (unscaled) {
            throw new IllegalStateException("unscale was already called for this step");
        }
        opt.setPreScale(Tensor.of(new float[] { 1.0f / scale }, 1));
        unscaled = true;
    }

    /** Unscales (unless done already) and records the optimiser step, skipped on the device on an overflow. */
    public void step(Optimizer opt) {
        if (!unscaled) {
            unscale(opt);
        }
        opt.step();
        last = opt;
    }

    /**
     * Adjusts the scale from this step's overflow flag (reading it waits for the step) and
     * returns whether the step's update was skipped.
     */
    public boolean update() {
        if (last == null || last.lastStats() == null) {
            throw new IllegalStateException("update() needs a step() first");
        }
        boolean overflow = last.lastStats().toFloatArray()[2] != 0.0f;
        if (overflow) {
            scale *= backoffFactor;
            cleanSteps = 0;
            last.rewindStep();
        } else if (++cleanSteps == growthInterval) {
            scale *= growthFactor;
            cleanSteps = 0;
        }
        last = null;
        unscaled = false;
        return overflow;
    }

    /** The scale and the count of clean steps since it last changed, for checkpoints. */
    public StateDict stateDict() {
        return new StateDict().put("scale", scale).put("cleanSteps", cleanSteps);
    }

    /** Restores the scale and step count saved by {@link #stateDict()}. */
    public void loadStateDict(StateDict d) {
        scale = d.floatValue("scale");
        cleanSteps = d.intValue("cleanSteps");
    }
}
