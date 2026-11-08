package ai.beestream.jitnn.dist;

import ai.beestream.jitnn.Tensor;

/**
 * Collective communication among the ranks of a group, each rank one thread and usually one
 * device (after torch.distributed). Every rank calls the same collectives in the same order with
 * tensors of the same shape; the calls block until all ranks have arrived.
 */
public interface Collective {

    enum ReduceOp {
        SUM, AVG, MAX, MIN
    }

    /** The number of ranks, over every process. */
    int worldSize();

    /** The global rank of this process's first rank (0 in a single process). */
    default int firstRank() {
        return 0;
    }

    /** The number of ranks in this process. */
    default int localRanks() {
        return worldSize();
    }

    /** Every rank's tensor replaced, in place, by the reduction over all ranks. */
    void allReduce(int rank, Tensor tensor, ReduceOp op);

    /** Every rank's tensor replaced, in place, by rank {@code root}'s. */
    void broadcast(int rank, Tensor tensor, int root);

    /**
     * input [worldSize * n, ...] on every rank; output [n, ...] receives on rank r the reduction of
     * every rank's r-th slice.
     */
    void reduceScatter(int rank, Tensor input, Tensor output, ReduceOp op);

    /** input [n, ...] on every rank; output [worldSize * n, ...] receives every rank's input, in rank order. */
    void allGather(int rank, Tensor input, Tensor output);

    /** Blocks until every rank has called it. */
    void barrier(int rank);

    /**
     * The reduction over all ranks of each tensor. A host collective reduces the tensors in place
     * and returns them; a device collective ({@link NcclCollective}) records the reduction and
     * returns new tensors, without waiting.
     */
    default Tensor[] allReduced(int rank, Tensor[] tensors, ReduceOp op) {
        for (Tensor t : tensors) {
            allReduce(rank, t, op);
        }
        return tensors;
    }

    /**
     * Starts the reduction over all ranks of {@code tensors}, packed into the persistent
     * {@code buffer} (exactly their total size), so that it runs while the operators recorded after
     * it do: NCCL runs it on a stream of its own. {@link #finishAllReduce} with the same
     * {@code tag} returns the results. Tags tell apart reductions in flight at the same time. A
     * collective that cannot overlap does nothing here and reduces in {@link #finishAllReduce}.
     */
    default void startAllReduce(int rank, Tensor[] tensors, Tensor buffer, ReduceOp op, int tag) {
    }

    /** The results of {@link #startAllReduce} (same tensors, buffer and tag), as {@link #allReduced} returns them. */
    default Tensor[] finishAllReduce(int rank, Tensor[] tensors, Tensor buffer, ReduceOp op, int tag) {
        return allReduced(rank, tensors, op);
    }

    /** Releases the ranks waiting in a collective after another rank failed. */
    default void abort() {
    }
}
