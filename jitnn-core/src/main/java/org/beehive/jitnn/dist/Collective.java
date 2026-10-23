package org.beehive.jitnn.dist;

import org.beehive.jitnn.Tensor;

/**
 * Collective communication among the ranks of a group, each rank one thread and usually one
 * device (after torch.distributed). Every rank calls the same collectives in the same order with
 * tensors of the same shape; the calls block until all ranks have arrived.
 */
public interface Collective {

    enum ReduceOp {
        SUM, AVG, MAX, MIN
    }

    /** The number of ranks. */
    int worldSize();

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

    /** Releases the ranks waiting in a collective after another rank failed. */
    default void abort() {
    }
}
