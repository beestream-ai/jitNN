package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/** Nearest or bilinear upsampling of NCHW by a scale factor, PyTorch's {@code nn.Upsample}. */
public final class Upsample extends Module {

    private final float scale;
    private final boolean bilinear;
    private final boolean alignCorners;

    public Upsample(float scale, boolean bilinear, boolean alignCorners) {
        this.scale = scale;
        this.bilinear = bilinear;
        this.alignCorners = alignCorners;
    }

    @Override
    public Tensor forward(Tensor x) {
        return bilinear ? F.upsampleBilinear(x, scale, alignCorners) : F.upsampleNearest(x, scale);
    }
}
