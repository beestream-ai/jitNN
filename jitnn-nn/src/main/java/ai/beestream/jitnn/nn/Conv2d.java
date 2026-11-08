package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * 2-D convolution over NCHW with a square kernel, PyTorch's {@code nn.Conv2d} semantics and
 * initialisation: weight and bias uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in =
 * in/groups * k * k.
 */
public final class Conv2d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int stride;
    private final int padding;
    private final int dilation;
    private final int groups;

    public Conv2d(int in, int out, int kernel, int stride, int padding, long seed) {
        this(in, out, kernel, stride, padding, 1, 1, true, seed);
    }

    public Conv2d(int in, int out, int kernel, int stride, int padding, int dilation, int groups, boolean bias, long seed) {
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        float bound = (float) (1.0 / Math.sqrt((double) in / groups * kernel * kernel));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, out, in / groups, kernel, kernel));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    public Tensor weight() {
        return weight;
    }

    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups);
    }

    /** relu(forward(x)), with the bias and ReLU fused into the convolution's output. */
    public Tensor forwardRelu(Tensor x) {
        return F.conv2dRelu(x, weight, bias, stride, padding, dilation, groups);
    }
}
