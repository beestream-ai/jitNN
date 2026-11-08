package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** Average pooling over 3-D windows, PyTorch's nn.AvgPool3d. */
public final class AvgPool3d extends Module {

    private final int kernel;
    private final int stride;
    private final int padding;

    /** Pools cubic windows of {@code kernel} with the given stride and padding. */
    public AvgPool3d(int kernel, int stride, int padding) {
        this.kernel = kernel;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.avgPool3d(x, kernel, stride, padding);
    }
}
