package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** Adaptive average pooling to fixed output sizes (1 to 3 spatial axes), PyTorch's nn.AdaptiveAvgPool1d/2d/3d. */
public final class AdaptiveAvgPool extends Module {

    private final int[] out;

    /** Pools to the output sizes {@code out}, one per spatial axis. */
    public AdaptiveAvgPool(int... out) {
        this.out = out.clone();
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.adaptiveAvgPool(x, out);
    }
}
