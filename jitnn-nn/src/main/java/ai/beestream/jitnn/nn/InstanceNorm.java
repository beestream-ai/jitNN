package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * InstanceNorm over [N, C, ...]: each sample and channel normalised over its positions, PyTorch's
 * nn.InstanceNorm1d/2d/3d without running statistics; affine optional (PyTorch's default: none).
 */
public final class InstanceNorm extends Module {

    private final float eps;
    private final Tensor weight;
    private final Tensor bias;

    public InstanceNorm(int channels, float eps, boolean affine) {
        this.eps = eps;
        this.weight = affine ? parameter(Tensor.ones(channels)) : null;
        this.bias = affine ? parameter(Tensor.zeros(channels)) : null;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.instanceNorm(x, weight, bias, eps);
    }
}
