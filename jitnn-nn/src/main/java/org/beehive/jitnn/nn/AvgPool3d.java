package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/** Average pooling over 3-D windows, PyTorch's nn.AvgPool3d. */
public final class AvgPool3d extends Module {

    private final int kernel;
    private final int stride;
    private final int padding;

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
