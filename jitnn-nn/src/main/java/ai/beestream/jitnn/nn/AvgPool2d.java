package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** Average pooling; padding counts in the divisor, as PyTorch's default. */
public final class AvgPool2d extends Module {

    private final int kernel;
    private final int stride;
    private final int padding;

    /** Pools non-overlapping {@code kernel} x {@code kernel} windows (stride {@code kernel}, no padding). */
    public AvgPool2d(int kernel) {
        this(kernel, kernel, 0);
    }

    /** Pools {@code kernel} x {@code kernel} windows with the given stride and padding. */
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
