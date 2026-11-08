package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** Adaptive max pooling to fixed output sizes (1 to 3 spatial axes), PyTorch's nn.AdaptiveMaxPool1d/2d/3d. */
public final class AdaptiveMaxPool extends Module {

    private final int[] out;

    /** Pools to the output sizes {@code out}, one per spatial axis. */
    public AdaptiveMaxPool(int... out) {
        this.out = out.clone();
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.adaptiveMaxPool(x, out);
    }
}
