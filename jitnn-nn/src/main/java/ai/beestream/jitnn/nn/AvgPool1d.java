package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** Average pooling over 1-D windows, PyTorch's nn.AvgPool1d. */
public final class AvgPool1d extends Module {

    private final int kernel;
    private final int stride;
    private final int padding;

    public AvgPool1d(int kernel, int stride, int padding) {
        this.kernel = kernel;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.avgPool1d(x, kernel, stride, padding);
    }
}
