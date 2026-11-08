package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * Batch normalisation over the channels of NCHW (or NC) input, PyTorch defaults: weight 1, bias
 * 0, running mean 0 and variance 1, momentum 0.1, eps 1e-5. Uses batch statistics in training
 * mode and the running statistics in evaluation mode.
 */
public final class BatchNorm2d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final Tensor runningMean;
    private final Tensor runningVar;
    private final float momentum;
    private final float eps;

    public BatchNorm2d(int channels) {
        this(channels, 0.1f, 1e-5f);
    }

    public BatchNorm2d(int channels, float momentum, float eps) {
        this.weight = parameter(Tensor.ones(channels));
        this.bias = parameter(Tensor.zeros(channels));
        this.runningMean = buffer(Tensor.zeros(channels));
        this.runningVar = buffer(Tensor.ones(channels));
        this.momentum = momentum;
        this.eps = eps;
    }

    public Tensor runningMean() {
        return runningMean;
    }

    public Tensor runningVar() {
        return runningVar;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.batchNorm(x, runningMean, runningVar, weight, bias, isTraining(), momentum, eps);
    }

    /** relu(forward(x)), fused into the normalisation kernel. */
    public Tensor forwardRelu(Tensor x) {
        return F.batchNormRelu(x, runningMean, runningVar, weight, bias, isTraining(), momentum, eps);
    }
}
