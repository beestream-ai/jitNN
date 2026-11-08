package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** [N, C, H, W] to [N, C], the mean over each feature map (adaptive average pooling to 1x1, flattened). */
public final class GlobalAvgPool2d extends Module {

    @Override
    public Tensor forward(Tensor x) {
        return F.globalAvgPool2d(x);
    }
}
