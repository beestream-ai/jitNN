package org.beehive.jitnn.nn;

import java.util.SplittableRandom;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * RReLU: random slopes in [lower, upper) while training (a fresh seed per call, so a replayed step
 * draws new slopes without recompiling), the mean slope in evaluation. PyTorch's defaults are 1/8
 * and 1/3.
 */
public final class RReLU extends Module {

    private final float lower;
    private final float upper;
    private final SplittableRandom seeds;

    public RReLU(float lower, float upper, long seed) {
        this.lower = lower;
        this.upper = upper;
        this.seeds = new SplittableRandom(seed);
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.rrelu(x, lower, upper, isTraining(), isTraining() ? Tensor.seed(seeds.nextLong()) : null);
    }
}
