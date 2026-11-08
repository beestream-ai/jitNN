package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * 1-D convolution over [N, C, ...] with a cubic kernel, optionally grouped, PyTorch's initialisation:
 * weight and bias uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in = in * kernel^1.
 */
public final class Conv1d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int stride;
    private final int padding;
    private final int dilation;
    private final int groups;

    public Conv1d(int in, int out, int kernel, int stride, int padding, int dilation, boolean bias, long seed) {
        this(in, out, kernel, stride, padding, dilation, 1, bias, seed);
    }

    /** With {@code groups} groups of in / groups input channels each (fan_in = in / groups * kernel^1). */
    public Conv1d(int in, int out, int kernel, int stride, int padding, int dilation, int groups, boolean bias, long seed) {
        if (in % groups != 0 || out % groups != 0) {
            throw new IllegalArgumentException(in + " and " + out + " channels do not split into " + groups + " groups");
        }
        this.groups = groups;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        float bound = (float) (1.0 / Math.sqrt(in / groups * Math.pow(kernel, 1)));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, out, in / groups, kernel));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.conv1d(x, weight, bias, stride, padding, dilation, groups);
    }
}
