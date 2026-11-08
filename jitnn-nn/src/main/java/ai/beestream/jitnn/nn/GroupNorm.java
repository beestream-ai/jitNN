package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** GroupNorm over [N, C, ...] with a per-channel affine (weight 1, bias 0 at the start), PyTorch's nn.GroupNorm. */
public final class GroupNorm extends Module {

    private final int groups;
    private final float eps;
    private final Tensor weight;
    private final Tensor bias;

    public GroupNorm(int groups, int channels, float eps, boolean affine) {
        this.groups = groups;
        this.eps = eps;
        this.weight = affine ? parameter(Tensor.ones(channels)) : null;
        this.bias = affine ? parameter(Tensor.zeros(channels)) : null;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.groupNorm(x, groups, weight, bias, eps);
    }
}
