package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

public final class MaxPool2d extends Module {

    private final int kernel;
    private final int stride;
    private final int padding;

    public MaxPool2d(int kernel) {
        this(kernel, kernel, 0);
    }

    public MaxPool2d(int kernel, int stride, int padding) {
        this.kernel = kernel;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.maxPool2d(x, kernel, stride, padding);
    }
}
