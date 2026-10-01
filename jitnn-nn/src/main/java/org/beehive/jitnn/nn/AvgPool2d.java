package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/** Average pooling; padding counts in the divisor, as PyTorch's default. */
public final class AvgPool2d extends Module {

    private final int kernel;
    private final int stride;
    private final int padding;

    public AvgPool2d(int kernel) {
        this(kernel, kernel, 0);
    }

    public AvgPool2d(int kernel, int stride, int padding) {
        this.kernel = kernel;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.avgPool2d(x, kernel, stride, padding);
    }
}
