package org.beehive.jitnn.dist;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.nccl.NcclCommunicator;

/**
 * Collectives on the GPUs, through NCCL (TornadoVM's tornado-nccl): one rank per CUDA device, in
 * one process. Unlike {@link HostCollective} the data never goes through the host, and the
 * collectives do not block: they are recorded into each rank's trace like any operator and run
 * inside the rank's plan (and its CUDA graph), so a training step's backward pass, gradient
 * all-reduce and optimiser step are one plan per rank, and the ranks' plans run at the same time.
 *
 * <p>Results are new tensors ({@link #allReduced}) or persistent tensors written in place
 * ({@link #allReduce}, {@link #broadcast}, the outputs of {@link #reduceScatter} and
 * {@link #allGather}): parameters, buffers and optimiser state. Gradients averaged with
 * {@link #allReduced} are packed into one buffer and reduced with one NCCL call.
 *
 * <pre>{@code
 * try (NcclCollective nccl = new NcclCollective("1:0", "1:1")) {
 *     ProcessGroup.run(new String[] { "1:0", "1:1" }, nccl, (rank, c) -> train(rank, c));
 * }
 * }</pre>
 *
 * <p>Close it after the plans that use it are released ({@link Engine#reset}), as NCCL waits for
 * CUDA graphs that still hold the communicator. FP32 tensors only.
 */
public final class NcclCollective implements Collective, AutoCloseable {

    private static final Map<Integer, NcclCommunicator> COMMUNICATORS = new ConcurrentHashMap<>();
    private static final AtomicInteger NEXT = new AtomicInteger(1);

    private final int id;
    private final NcclCommunicator communicator;
    private final int world;
    private final Map<Integer, Tensor> barrierTensors = new ConcurrentHashMap<>();

    /** A communicator over {@code devices} ("driver:device" as for {@code tornado --devices}), rank r on devices[r]: distinct CUDA GPUs. */
    public NcclCollective(String... devices) {
        TornadoDevice[] d = new TornadoDevice[devices.length];
        for (int i = 0; i < devices.length; i++) {
            String[] parts = devices[i].trim().split(":");
            d[i] = TornadoExecutionPlan.getDevice(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
            if (d[i].getTornadoVMBackend() != TornadoVMBackendType.CUDA) {
                throw new IllegalArgumentException("NCCL needs CUDA devices; " + devices[i] + " is " + d[i].getTornadoVMBackend());
            }
        }
        this.world = devices.length;
        this.communicator = NcclCommunicator.create(d);
        this.id = NEXT.getAndIncrement();
        COMMUNICATORS.put(id, communicator);
    }

    static NcclCommunicator communicator(int id) {
        NcclCommunicator c = COMMUNICATORS.get(id);
        if (c == null) {
            throw new IllegalStateException("NCCL communicator " + id + " is closed");
        }
        return c;
    }

    @Override
    public int worldSize() {
        return world;
    }

    /** Persistent {@code tensor} replaced, on every rank, by the reduction over the ranks. */
    @Override
    public void allReduce(int rank, Tensor tensor, ReduceOp op) {
        persistent(tensor, "allReduce");
        NcclOp.inPlace(NcclOp.ALL_REDUCE, id, tensor, op.ordinal());
    }

    /** Persistent {@code tensor} replaced, on every rank, by rank {@code root}'s. */
    @Override
    public void broadcast(int rank, Tensor tensor, int root) {
        persistent(tensor, "broadcast");
        NcclOp.inPlace(NcclOp.BROADCAST, id, tensor, root);
    }

    /** The reduction over the ranks of each tensor, as new tensors: one NCCL call for all of them. */
    @Override
    public Tensor[] allReduced(int rank, Tensor[] tensors, ReduceOp op) {
        for (Tensor t : tensors) {
            f32(t, "allReduced");
        }
        return tensors.length == 0 ? tensors : NcclOp.bucket(id, tensors, op.ordinal());
    }

    /** Persistent {@code output} [n, ...] receives the reduction of every rank's slice r of input [world * n, ...]. */
    @Override
    public void reduceScatter(int rank, Tensor input, Tensor output, ReduceOp op) {
        persistent(output, "reduceScatter");
        f32(input, "reduceScatter");
        if (input.numel() != world * output.numel()) {
            throw new IllegalArgumentException("reduceScatter: input of " + input.numel() + " values for an output of " + output.numel() + " on " + world + " ranks");
        }
        NcclOp.into(NcclOp.REDUCE_SCATTER, id, input, output, op.ordinal());
    }

    /** Persistent {@code output} [world * n, ...] receives every rank's input [n, ...], in rank order. */
    @Override
    public void allGather(int rank, Tensor input, Tensor output) {
        persistent(output, "allGather");
        f32(input, "allGather");
        if (output.numel() != world * input.numel()) {
            throw new IllegalArgumentException("allGather: output of " + output.numel() + " values for an input of " + input.numel() + " on " + world + " ranks");
        }
        NcclOp.into(NcclOp.ALL_GATHER, id, input, output, 0);
    }

    /** Runs everything recorded so far, and returns once every rank has. */
    @Override
    public void barrier(int rank) {
        Tensor t = barrierTensors.computeIfAbsent(rank, r -> Tensor.persistent(Tensor.zeros(1)));
        allReduce(rank, t, ReduceOp.SUM);
        Engine.sync(t);
    }

    /** Releases ranks waiting in a collective after another rank failed; the communicator is unusable after it. */
    @Override
    public void abort() {
        communicator.abort();
    }

    @Override
    public void close() {
        COMMUNICATORS.remove(id);
        if (!communicator.isAborted()) {
            communicator.close();
        }
    }

    private static void f32(Tensor t, String what) {
        if (t.dtype() != DType.F32) {
            throw new IllegalArgumentException(what + ": NCCL collectives take FP32 tensors, got " + t);
        }
    }

    private static void persistent(Tensor t, String what) {
        f32(t, what);
        if (!t.storage().isPersistent()) {
            throw new IllegalArgumentException(what + " writes in place, so it needs a persistent tensor (a parameter, buffer or Tensor.persistent); use allReduced for others: " + t);
        }
    }
}
