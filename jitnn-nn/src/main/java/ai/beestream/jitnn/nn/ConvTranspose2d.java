package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * Transposed 2-D convolution over NCHW with a square kernel, PyTorch's {@code nn.ConvTranspose2d}
 * semantics and initialisation: weight [in, out/groups, k, k] and bias uniform in
 * [-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in = out/groups * k * k.
 */
public final class ConvTranspose2d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int stride;
    private final int padding;
    private final int outputPadding;
    private final int dilation;
    private final int groups;

    public ConvTranspose2d(int in, int out, int kernel, int stride, int padding, long seed) {
        this(in, out, kernel, stride, padding, 0, 1, 1, true, seed);
    }

    public ConvTranspose2d(int in, int out, int kernel, int stride, int padding, int outputPadding, int dilation, int groups, boolean bias, long seed) {
        this.stride = stride;
        this.padding = padding;
        this.outputPadding = outputPadding;
        this.dilation = dilation;
        this.groups = groups;
        float bound = (float) (1.0 / Math.sqrt((double) out / groups * kernel * kernel));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, in, out / groups, kernel, kernel));
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
        return F.convTranspose2d(x, weight, bias, stride, padding, outputPadding, dilation, groups);
    }
}
