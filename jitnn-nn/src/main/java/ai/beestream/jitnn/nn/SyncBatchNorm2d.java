package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.dist.Collective;
import ai.beestream.jitnn.dist.NcclCollective;
import ai.beestream.jitnn.ops.Assign;

/**
 * Batch normalisation whose training statistics are those of the batch over all data-parallel
 * ranks, as PyTorch's SyncBatchNorm: each rank sums its x and x^2 per channel, one all-reduce
 * gives the global mean and variance, and the backward pass all-reduces their gradients the same
 * way. Ranks training on shares of a batch then match one device training on the whole batch.
 * Evaluation uses the running statistics, which are updated alike on every rank. Needs an
 * {@link NcclCollective} (its all-reduce is differentiable); without a collective it is plain
 * batch normalisation. NCHW input, PyTorch defaults (momentum 0.1, eps 1e-5).
 */
public final class SyncBatchNorm2d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final Tensor runningMean;
    private final Tensor runningVar;
    private final float momentum;
    private final float eps;
    private final int channels;
    private final Collective collective;
    private final int rank;

    /**
     * Synchronised batch normalisation for rank {@code rank}, with PyTorch's defaults (momentum 0.1, eps
     * 1e-5).
     */
    public SyncBatchNorm2d(int channels, Collective collective, int rank) {
        this(channels, collective, rank, 0.1f, 1e-5f);
    }

    /**
     * Synchronised batch normalisation for rank {@code rank} of an
     * {@link ai.beestream.jitnn.dist.NcclCollective}; with a null collective (or one rank) it is plain batch
     * normalisation.
     */
    public SyncBatchNorm2d(int channels, Collective collective, int rank, float momentum, float eps) {
        if (collective != null && !(collective instanceof NcclCollective)) {
            throw new IllegalArgumentException("SyncBatchNorm2d needs an NcclCollective (a differentiable all-reduce), got " + collective.getClass().getSimpleName());
        }
        this.channels = channels;
        this.collective = collective;
        this.rank = rank;
        this.weight = parameter(Tensor.ones(channels));
        this.bias = parameter(Tensor.zeros(channels));
        this.runningMean = buffer(Tensor.zeros(channels));
        this.runningVar = buffer(Tensor.ones(channels));
        this.momentum = momentum;
        this.eps = eps;
    }

    /** The running mean [channels], updated in training mode. */
    public Tensor runningMean() {
        return runningMean;
    }

    /** The running variance [channels], updated in training mode. */
    public Tensor runningVar() {
        return runningVar;
    }

    /** Sum over the batch and spatial axes of x [N, C, H, W]: [C]. */
    private static Tensor channelSum(Tensor x) {
        return x.sum(0, false).sum(1, false).sum(1, false);
    }

    @Override
    public Tensor forward(Tensor x) {
        if (!isTraining() || collective == null || collective.worldSize() == 1) {
            return F.batchNorm(x, runningMean, runningVar, weight, bias, isTraining(), momentum, eps);
        }
        if (x.rank() != 4 || x.dim(1) != channels) {
            throw new IllegalArgumentException("SyncBatchNorm2d(" + channels + ") on " + x.shape());
        }
        int c = channels;
        float count = (float) x.dim(0) * x.dim(2) * x.dim(3) * collective.worldSize();
        Tensor local = Tensor.cat(0, channelSum(x), channelSum(x.square()));
        Tensor global = collective.allReduced(rank, new Tensor[] { local }, Collective.ReduceOp.SUM)[0];
        Tensor mean = global.narrow(0, 0, c).mul(1.0f / count);
        Tensor var = global.narrow(0, c, c).mul(1.0f / count).sub(mean.square());
        Tensor inv = var.add(eps).sqrt();
        Tensor y = x.sub(mean.reshape(1, c, 1, 1)).div(inv.reshape(1, c, 1, 1)).mul(weight.reshape(1, c, 1, 1)).add(bias.reshape(1, c, 1, 1));
        try (Grad.Scope off = Grad.disabled()) {
            // the running variance is the unbiased one, as in PyTorch
            Assign.into(runningMean.mul(1 - momentum).add(mean.detach().mul(momentum)), runningMean);
            Assign.into(runningVar.mul(1 - momentum).add(var.detach().mul(momentum * count / Math.max(count - 1, 1))), runningVar);
        }
        return y;
    }
}
