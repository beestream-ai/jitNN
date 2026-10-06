package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/** Adaptive average pooling to fixed output sizes (1 to 3 spatial axes), PyTorch's nn.AdaptiveAvgPool1d/2d/3d. */
public final class AdaptiveAvgPool extends Module {

    private final int[] out;

    public AdaptiveAvgPool(int... out) {
        this.out = out.clone();
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.adaptiveAvgPool(x, out);
    }
}
