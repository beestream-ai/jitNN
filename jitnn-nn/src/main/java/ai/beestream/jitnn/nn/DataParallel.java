package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.dist.Collective;

/**
 * Data-parallel training with one model replica per rank (see
 * {@link ai.beestream.jitnn.dist.ProcessGroup}): broadcast rank 0's parameters once, then after
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
            if (b.dtype() == ai.beestream.jitnn.DType.F32) {
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
        java.util.List<Tensor> rest = new java.util.ArrayList<>();
        if (buckets != null) {
            for (Bucket b : buckets) {
                if (b.started) {
                    Tensor[] grads = new Tensor[b.params.size()];
                    for (int i = 0; i < grads.length; i++) {
                        grads[i] = b.params.get(i).grad();
                    }
                    Tensor[] mean = collective.finishAllReduce(rank, grads, b.buffer, Collective.ReduceOp.AVG, b.tag);
                    for (int i = 0; i < mean.length; i++) {
                        b.params.get(i).setGrad(mean[i]);
                    }
                } else {
                    // a bucket with parameters that got no gradient this step: reduced with the rest
                    rest.addAll(b.params);
                }
                b.ready = 0;
                b.started = false;
            }
        } else {
            rest.addAll(model.parameters());
        }
        java.util.List<Tensor> params = new java.util.ArrayList<>();
        java.util.List<Tensor> grads = new java.util.ArrayList<>();
        for (Tensor p : rest) {
            if (p.grad() != null) {
                params.add(p);
                grads.add(p.grad());
            }
        }
        if (params.isEmpty()) {
            return;
        }
        Tensor[] mean = collective.allReduced(rank, grads.toArray(new Tensor[0]), Collective.ReduceOp.AVG);
        for (int i = 0; i < mean.length; i++) {
            params.get(i).setGrad(mean[i]);
        }
    }

    /** Parameters whose gradients are reduced together, in the order they become ready. */
    private static final class Bucket {
        final java.util.List<Tensor> params = new java.util.ArrayList<>();
        Tensor buffer;
        int tag;
        int ready;
        boolean started;
    }

    private java.util.List<Bucket> buckets;

    /**
     * Overlaps the gradient all-reduce with the backward pass, as PyTorch's DistributedDataParallel:
     * the parameters are grouped, in reverse order (the order their gradients become final), into
     * buckets of about {@code bucketBytes}; a bucket's all-reduce starts as soon as its last gradient
     * is final, while the backward pass of the earlier layers goes on, and {@link #averageGradients}
     * only waits for it. With NCCL the reductions run on a stream of their own; other collectives
     * reduce at {@link #averageGradients} as before. One backward pass per step.
     */
    public DataParallel overlapGradients(long bucketBytes) {
        java.util.List<Tensor> params = new java.util.ArrayList<>(model.parameters());
        java.util.Collections.reverse(params);
        buckets = new java.util.ArrayList<>();
        Bucket b = null;
        long bytes = 0;
        for (Tensor p : params) {
            if (b == null || bytes >= bucketBytes) {
                b = new Bucket();
                b.tag = buckets.size() + 1;
                buckets.add(b);
                bytes = 0;
            }
            b.params.add(p);
            bytes += 4L * p.numel();
            Bucket owner = b;
            p.onGradReady(t -> ready(owner));
        }
        for (Bucket k : buckets) {
            int total = 0;
            for (Tensor p : k.params) {
                total += p.numel();
            }
            k.buffer = Tensor.persistent(Tensor.zeros(total));
        }
        return this;
    }

    private void ready(Bucket b) {
        if (++b.ready == b.params.size()) {
            Tensor[] grads = new Tensor[b.params.size()];
            for (int i = 0; i < grads.length; i++) {
                grads[i] = b.params.get(i).grad();
            }
            collective.startAllReduce(rank, grads, b.buffer, Collective.ReduceOp.AVG, b.tag);
            b.started = true;
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
