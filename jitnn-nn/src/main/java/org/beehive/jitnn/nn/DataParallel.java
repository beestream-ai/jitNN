package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.dist.Collective;

/**
 * Data-parallel training with one model replica per rank (see
 * {@link org.beehive.jitnn.dist.ProcessGroup}): broadcast rank 0's parameters once, then after
 * each backward pass average the gradients over the ranks before the optimiser step, so every
 * replica takes the same step, as PyTorch's DistributedDataParallel does.
 */
public final class DataParallel {

    private final Module model;
    private final Collective collective;
    private final int rank;

    public DataParallel(Module model, int rank, Collective collective) {
        this.model = model;
        this.rank = rank;
        this.collective = collective;
    }

    /** Copies rank 0's parameters and buffers to every rank. */
    public DataParallel broadcastParameters() {
        for (Tensor p : model.parameters()) {
            collective.broadcast(rank, p, 0);
        }
        for (Tensor b : model.buffers()) {
            if (b.dtype() == org.beehive.jitnn.DType.F32) {
                collective.broadcast(rank, b, 0);
            }
        }
        return this;
    }

    /**
     * Replaces each parameter's gradient by its mean over the ranks (parameters without a gradient
     * are skipped on every rank alike), with one collective for all of them.
     */
    public void averageGradients() {
        java.util.List<Tensor> params = new java.util.ArrayList<>();
        java.util.List<Tensor> grads = new java.util.ArrayList<>();
        for (Tensor p : model.parameters()) {
            if (p.grad() != null) {
                params.add(p);
                grads.add(p.grad());
            }
        }
        Tensor[] mean = collective.allReduced(rank, grads.toArray(new Tensor[0]), Collective.ReduceOp.AVG);
        for (int i = 0; i < mean.length; i++) {
            params.get(i).setGrad(mean[i]);
        }
    }

    public Module module() {
        return model;
    }

    public int rank() {
        return rank;
    }

    public int worldSize() {
        return collective.worldSize();
    }
}
