package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** Max pooling over 1-D windows, PyTorch's nn.MaxPool1d. */
public final class MaxPool1d extends Module {

    private final int kernel;
    private final int stride;
    private final int padding;

    /** Pools windows of {@code kernel} with the given stride and padding. */
    public MaxPool1d(int kernel, int stride, int padding) {
        this.kernel = kernel;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.maxPool1d(x, kernel, stride, padding);
    }
}
