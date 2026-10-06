package org.beehive.jitnn.dist;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import org.beehive.jitnn.Tensor;

/**
 * Collectives staged through host memory: each rank copies its tensor to the host, the reduction
 * runs on the host in rank order (so every rank gets bit-identical results, run after run), and
 * each rank copies the result back. It works with any mix of devices and backends (CUDA, OpenCL,
 * Metal, even different backends in one group); the price is a device-host round trip per call.
 */
public final class HostCollective implements Collective {

    private final int world;
    private final CyclicBarrier barrier;
    private final float[][] staged;
    private volatile float[] result;

    public HostCollective(int worldSize) {
        if (worldSize < 1) {
            throw new IllegalArgumentException("world size " + worldSize);
        }
        this.world = worldSize;
        this.barrier = new CyclicBarrier(worldSize);
        this.staged = new float[worldSize][];
    }

    @Override
    public int worldSize() {
        return world;
    }

    private void await() {
        try {
            barrier.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("collective interrupted", e);
        } catch (BrokenBarrierException e) {
            throw new IllegalStateException("another rank failed during a collective", e);
        }
    }

    private void check(int rank) {
        if (rank < 0 || rank >= world) {
            throw new IllegalArgumentException("rank " + rank + " outside [0, " + world + ")");
        }
    }

    /** Element-wise reduction of the staged arrays, in rank order. */
    private float[] reduce(ReduceOp op) {
        int n = staged[0].length;
        for (int r = 1; r < world; r++) {
            if (staged[r].length != n) {
                throw new IllegalArgumentException("rank " + r + " has " + staged[r].length + " elements, rank 0 has " + n);
            }
        }
        float[] out = staged[0].clone();
        for (int r = 1; r < world; r++) {
            float[] v = staged[r];
            for (int i = 0; i < n; i++) {
                out[i] = switch (op) {
                    case SUM, AVG -> out[i] + v[i];
                    case MAX -> Math.max(out[i], v[i]);
                    case MIN -> Math.min(out[i], v[i]);
                };
            }
        }
        if (op == ReduceOp.AVG) {
            float inv = 1.0f / world;
            for (int i = 0; i < n; i++) {
                out[i] *= inv;
            }
        }
        return out;
    }

    @Override
    public void allReduce(int rank, Tensor tensor, ReduceOp op) {
        check(rank);
        staged[rank] = tensor.toFloatArray();
        await();
        if (rank == 0) {
            result = reduce(op);
        }
        await();
        tensor.copyFrom(result);
        await();
    }

    @Override
    public void broadcast(int rank, Tensor tensor, int root) {
        check(rank);
        if (rank == root) {
            result = tensor.toFloatArray();
        }
        await();
        if (rank != root) {
            tensor.copyFrom(result);
        }
        await();
    }

    @Override
    public void reduceScatter(int rank, Tensor input, Tensor output, ReduceOp op) {
        check(rank);
        staged[rank] = input.toFloatArray();
        await();
        if (rank == 0) {
            result = reduce(op);
        }
        await();
        int n = output.numel();
        if (result.length != n * world) {
            throw new IllegalArgumentException("reduce-scatter input of " + result.length + " elements for " + world + " outputs of " + n);
        }
        output.copyFrom(java.util.Arrays.copyOfRange(result, rank * n, (rank + 1) * n));
        await();
    }

    @Override
    public void allGather(int rank, Tensor input, Tensor output) {
        check(rank);
        staged[rank] = input.toFloatArray();
        await();
        int n = staged[rank].length;
        float[] all = new float[n * world];
        for (int r = 0; r < world; r++) {
            System.arraycopy(staged[r], 0, all, r * n, n);
        }
        output.copyFrom(all);
        await();
    }

    /** Releases every rank waiting in a collective with an error (called when a rank fails). */
    public void abort() {
        barrier.reset();
    }

    @Override
    public void barrier(int rank) {
        check(rank);
        await();
    }
}
