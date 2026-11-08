package ai.beestream.jitnn.dist;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Native.Lib;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.nccl.Nccl;
import uk.ac.manchester.tornado.nccl.NcclCommunicator;
import uk.ac.manchester.tornado.nccl.NcclRedOp;

/**
 * The NCCL collectives of {@link NcclCollective} as an operator, so they run inside the rank's plan.
 * iattr = [kind, communicator id, reduction (a ReduceOp ordinal) or broadcast root].
 */
final class NcclOp implements Op {

    static final int ALL_REDUCE = 0, BROADCAST = 1, BUCKET = 2, ALL_GATHER = 3, REDUCE_SCATTER = 4, START = 5, FINISH = 6;

    static final NcclOp OP = new NcclOp();

    private NcclOp() {
    }

    /** A persistent tensor reduced or broadcast in place. */
    static void inPlace(int kind, int comm, Tensor t, int arg) {
        Tensor.record(OP, new Tensor[] { t }, new Tensor[] { t }, new int[] { kind, comm, arg }, null, true);
    }

    /** The tensors all-reduced into new tensors, through one packed buffer. */
    static Tensor[] bucket(int comm, Tensor[] in, int op) {
        Tensor[] out = new Tensor[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = Tensor.result(in[i].shape(), in[i].dtype());
        }
        Tensor.record(OP, in.clone(), out, new int[] { BUCKET, comm, op }, null, true);
        return out;
    }

    /**
     * Packs the tensors into persistent {@code buffer} and starts its all-reduce on NCCL's side stream.
     * inputs [tensors..., buffer], outputs [buffer]; iattr = [START, comm, op, tag].
     */
    static void start(int comm, Tensor[] in, Tensor buffer, int op, int tag) {
        Tensor[] inputs = java.util.Arrays.copyOf(in, in.length + 1);
        inputs[in.length] = buffer;
        Tensor.record(OP, inputs, new Tensor[] { buffer }, new int[] { START, comm, op, tag }, null, true);
    }

    /** Waits for the all-reduce started on {@code buffer} with {@code tag}; unpacks it into new tensors shaped like {@code like}. */
    static Tensor[] finish(int comm, Tensor[] like, Tensor buffer, int tag) {
        Tensor[] out = new Tensor[like.length];
        for (int i = 0; i < like.length; i++) {
            out[i] = Tensor.result(like[i].shape(), like[i].dtype());
        }
        Tensor.record(OP, new Tensor[] { buffer }, out, new int[] { FINISH, comm, 0, tag }, null, true);
        return out;
    }

    /** input into a persistent output (all-gather, reduce-scatter). */
    static void into(int kind, int comm, Tensor input, Tensor output, int arg) {
        Tensor.record(OP, new Tensor[] { input }, new Tensor[] { output }, new int[] { kind, comm, arg }, null, true);
    }

    private static NcclRedOp redOp(int op) {
        return switch (Collective.ReduceOp.values()[op]) {
            case SUM -> NcclRedOp.SUM;
            case AVG -> NcclRedOp.AVG;
            case MAX -> NcclRedOp.MAX;
            case MIN -> NcclRedOp.MIN;
        };
    }

    public static void pack(KernelContext kc, FloatArray src, FloatArray flat, int offset, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            flat.set(offset + i, src.get(i));
        }
    }

    public static void unpack(KernelContext kc, FloatArray flat, FloatArray dst, int offset, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            dst.set(i, flat.get(offset + i));
        }
    }

    @Override
    public String name() {
        return "nccl";
    }

    @Override
    public void emit(Emitter e, Node n) {
        int kind = n.iattr[0];
        NcclCommunicator comm = NcclCollective.communicator(n.iattr[1]);
        int arg = n.iattr[2];
        switch (kind) {
            case ALL_REDUCE -> {
                FloatArray buf = e.f32(n.out(0));
                NcclRedOp op = redOp(arg);
                e.library(Lib.NCCL, "allReduce", (g, id) -> g.libraryTask(id, Nccl::allReduceInPlace, comm, buf, op));
            }
            case BROADCAST -> {
                FloatArray buf = e.f32(n.out(0));
                e.library(Lib.NCCL, "broadcast", (g, id) -> g.libraryTask(id, Nccl::broadcast, comm, buf, arg));
            }
            case BUCKET -> {
                // NCCL reduces whole arrays and jitNN's intermediates may sit in larger reused buffers:
                // pack the tensors into one buffer of exactly their size, reduce it, unpack
                int total = 0;
                for (Tensor t : n.inputs) {
                    total += t.numel();
                }
                FloatArray flat = e.scratch(total);
                int offset = 0;
                for (int i = 0; i < n.inputs.length; i++) {
                    FloatArray src = e.f32(n.in(i));
                    int count = n.in(i).numel();
                    int at = offset;
                    e.task(Emitter.grid1D(count), (g, id) -> g.task(id, NcclOp::pack, e.context(), src, flat, at, count));
                    offset += count;
                }
                NcclRedOp op = redOp(arg);
                e.library(Lib.NCCL, "allReduce", (g, id) -> g.libraryTask(id, Nccl::allReduceInPlace, comm, flat, op));
                offset = 0;
                for (int i = 0; i < n.outputs.length; i++) {
                    FloatArray dst = e.f32(n.out(i));
                    int count = n.out(i).numel();
                    int at = offset;
                    e.task(Emitter.grid1D(count), (g, id) -> g.task(id, NcclOp::unpack, e.context(), flat, dst, at, count));
                    offset += count;
                }
            }
            case START -> {
                // the buffer is persistent, so exactly the size every rank reduces
                FloatArray flat = e.f32(n.in(n.inputs.length - 1));
                int offset = 0;
                for (int i = 0; i < n.inputs.length - 1; i++) {
                    FloatArray src = e.f32(n.in(i));
                    int count = n.in(i).numel();
                    int at = offset;
                    e.task(Emitter.grid1D(count), (g, id) -> g.task(id, NcclOp::pack, e.context(), src, flat, at, count));
                    offset += count;
                }
                NcclRedOp op = redOp(arg);
                int tag = n.iattr[3];
                e.libraryForked(Lib.NCCL, "allReduceStart", (g, id) -> g.libraryTask(id, Nccl::allReduceInPlaceStart, comm, flat, op, tag), "nccl" + n.iattr[1] + ":" + tag,
                        (g, id) -> g.libraryTask(id, Nccl::allReduceWait, comm, flat, tag));
            }
            case FINISH -> {
                FloatArray flat = e.f32(n.in(0));
                int tag = n.iattr[3];
                e.libraryJoin(Lib.NCCL, "allReduceWait", (g, id) -> g.libraryTask(id, Nccl::allReduceWait, comm, flat, tag), "nccl" + n.iattr[1] + ":" + tag);
                int offset = 0;
                for (int i = 0; i < n.outputs.length; i++) {
                    FloatArray dst = e.f32(n.out(i));
                    int count = n.out(i).numel();
                    int at = offset;
                    e.task(Emitter.grid1D(count), (g, id) -> g.task(id, NcclOp::unpack, e.context(), flat, dst, at, count));
                    offset += count;
                }
            }
            case ALL_GATHER, REDUCE_SCATTER -> {
                FloatArray src = e.f32(n.in(0));
                int count = n.in(0).numel();
                FloatArray send = src.getSize() == count ? src : e.scratch(count);
                if (send != src) {
                    e.task(Emitter.grid1D(count), (g, id) -> g.task(id, NcclOp::pack, e.context(), src, send, 0, count));
                }
                FloatArray recv = e.f32(n.out(0));
                if (kind == ALL_GATHER) {
                    e.library(Lib.NCCL, "allGather", (g, id) -> g.libraryTask(id, Nccl::allGather, comm, send, recv));
                } else {
                    NcclRedOp op = redOp(arg);
                    e.library(Lib.NCCL, "reduceScatter", (g, id) -> g.libraryTask(id, Nccl::reduceScatter, comm, send, recv, op));
                }
            }
            default -> throw new IllegalStateException("nccl kind " + kind);
        }
    }

    /**
     * The gradient of an all-reduce of one tensor (SUM or AVG) is the same all-reduce of the
     * output's gradient on every rank, recorded on each rank's backward pass alike.
     */
    @Override
    public Tensor[] backward(Node n, Tensor gradOut) {
        int op = n.iattr[2];
        boolean linear = op == Collective.ReduceOp.SUM.ordinal() || op == Collective.ReduceOp.AVG.ordinal();
        if (n.iattr[0] != BUCKET || n.inputs.length != 1 || !linear) {
            throw new UnsupportedOperationException("only a SUM or AVG all-reduce of one tensor has a gradient");
        }
        return new Tensor[] { bucket(n.iattr[1], new Tensor[] { gradOut }, op)[0] };
    }

    @Override
    public void reference(Node n) {
        throw new UnsupportedOperationException("NCCL collectives run on CUDA devices (TornadoVM backend), not on the host reference backend");
    }
}
