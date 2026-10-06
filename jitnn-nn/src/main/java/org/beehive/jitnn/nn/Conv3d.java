package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * 3-D convolution over [N, C, ...] with a cubic kernel and groups 1, PyTorch's initialisation:
 * weight and bias uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in = in * kernel^3.
 */
public final class Conv3d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int stride;
    private final int padding;
    private final int dilation;

    public Conv3d(int in, int out, int kernel, int stride, int padding, int dilation, boolean bias, long seed) {
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        float bound = (float) (1.0 / Math.sqrt(in * Math.pow(kernel, 3)));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, out, in, kernel, kernel, kernel));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.conv3d(x, weight, bias, stride, padding, dilation);
    }
}
