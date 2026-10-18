package org.beehive.jitnn.engine;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.beehive.jitnn.Tensor;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * What an operator sees while its device tasks are added to a task graph: the arrays bound to
 * its tensors, constant metadata arrays, and helpers for launch grids.
 */
public final class Emitter {

    /** Adds one task to the graph under the given task id. */
    @FunctionalInterface
    public interface TaskAdder {
        void add(TaskGraph graph, String taskId);
    }

    /** Work-group size for 1-D kernels and group reductions. Kernels assume this value. */
    public static final int LOCAL = 256;

    record Spec(String id, WorkerGrid grid, TaskAdder adder, int node) {
    }

    private String graphName;
    private int node = -1;
    /** Per node, the arrays its tasks use (identity), in order of first use. */
    final List<List<Object>> uses = new ArrayList<>();

    /** Starts recording the tasks and arrays of the next node. */
    void beginNode() {
        node++;
        uses.add(new ArrayList<>());
    }

    private <T> T use(T array) {
        if (node >= 0) {
            List<Object> u = uses.get(node);
            for (Object o : u) {
                if (o == array) {
                    return array;
                }
            }
            u.add(array);
        }
        return array;
    }
    private final Map<Storage, Object> arrays;
    final List<Spec> tasks = new ArrayList<>();
    final List<Object> constants = new ArrayList<>();
    final List<Object> scratch = new ArrayList<>();
    private final Map<String, IntArray> constantCache = new java.util.HashMap<>();

    private final java.util.Set<Storage> unread;

    Emitter(String graphName, IdentityHashMap<Storage, Object> arrays, java.util.Set<Storage> unread) {
        this.graphName = graphName;
        this.arrays = arrays;
        this.unread = unread;
    }

    /**
     * True when nothing reads {@code t} after this plan: an internal output (no user code holds it)
     * that no later operator of the trace consumes and that was not requested. An operator may then
     * leave it unwritten.
     */
    public boolean unread(Tensor t) {
        return unread.contains(t.storage());
    }

    public FloatArray f32(Tensor t) {
        return use((FloatArray) arrays.get(t.storage()));
    }

    public IntArray i32(Tensor t) {
        return use((IntArray) arrays.get(t.storage()));
    }

    public uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray f16(Tensor t) {
        return use((uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray) arrays.get(t.storage()));
    }

    /** Q8_0 tensors: GGUF's 34-byte blocks of 32 values. */
    public uk.ac.manchester.tornado.api.types.arrays.ByteArray q8(Tensor t) {
        return use((uk.ac.manchester.tornado.api.types.arrays.ByteArray) arrays.get(t.storage()));
    }

    /** GGUF block-quantised tensors: their raw blocks. */
    public uk.ac.manchester.tornado.api.types.arrays.ByteArray blocks(Tensor t) {
        return use((uk.ac.manchester.tornado.api.types.arrays.ByteArray) arrays.get(t.storage()));
    }

    /** FP8 tensors: one byte per value. */
    public uk.ac.manchester.tornado.api.types.arrays.ByteArray fp8(Tensor t) {
        return use((uk.ac.manchester.tornado.api.types.arrays.ByteArray) arrays.get(t.storage()));
    }

    /** bfloat16 tensors are stored as their raw 16-bit patterns. */
    public uk.ac.manchester.tornado.api.types.arrays.ShortArray bf16(Tensor t) {
        return use((uk.ac.manchester.tornado.api.types.arrays.ShortArray) arrays.get(t.storage()));
    }

    /** A constant int array uploaded once with the graph (shape and stride metadata). */
    public IntArray constant(int... values) {
        String key = java.util.Arrays.toString(values);
        return use(constantCache.computeIfAbsent(key, k -> {
            IntArray a = IntArray.fromArray(values.length == 0 ? new int[] { 0 } : values);
            constants.add(a);
            return a;
        }));
    }

    private final Map<String, FloatArray> constantF32Cache = new java.util.HashMap<>();

    /** A constant float array uploaded once with the graph. */
    public FloatArray constantF32(float... values) {
        String key = java.util.Arrays.toString(values);
        return use(constantF32Cache.computeIfAbsent(key, k -> {
            FloatArray a = FloatArray.fromArray(values.length == 0 ? new float[] { 0 } : values);
            constants.add(a);
            return a;
        }));
    }

    private FloatArray dummy;

    /** A one-element array for optional array arguments that a kernel does not read. */
    public FloatArray dummyF32() {
        if (dummy == null) {
            dummy = new FloatArray(1);
            constants.add(dummy);
        }
        return use(dummy);
    }

    /** A device work buffer owned by the compiled plan (partial results between two passes). */
    public FloatArray scratch(int numel) {
        FloatArray a = new FloatArray(Math.max(numel, 1));
        scratch.add(a);
        return use(a);
    }

    private final java.util.Map<String, Object> shared = new java.util.HashMap<>();

    /**
     * A work buffer of exactly {@code numel} elements shared by every task of the plan that asks for
     * the same role and size: tasks run in order, so a buffer one operator fills and reads is free for
     * the next (the layout copies around library calls, which would otherwise take a buffer each).
     */
    public FloatArray sharedScratch(String role, int numel) {
        return use((FloatArray) shared.computeIfAbsent(role + ":f32:" + numel, k -> {
            FloatArray a = new FloatArray(Math.max(numel, 1));
            scratch.add(a);
            return a;
        }));
    }

    /** {@link #sharedScratch} for FP16 buffers. */
    public uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray sharedScratchF16(String role, int numel) {
        return use((uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray) shared.computeIfAbsent(role + ":f16:" + numel, k -> {
            var a = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(Math.max(numel, 1));
            scratch.add(a);
            return a;
        }));
    }

    /** An FP16 device work buffer owned by the compiled plan. */
    public uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray scratchF16(int numel) {
        var a = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(Math.max(numel, 1));
        scratch.add(a);
        return use(a);
    }

    /** An INT32 device work buffer owned by the compiled plan. */
    public uk.ac.manchester.tornado.api.types.arrays.IntArray scratchI32(int numel) {
        uk.ac.manchester.tornado.api.types.arrays.IntArray a = new uk.ac.manchester.tornado.api.types.arrays.IntArray(Math.max(numel, 1));
        scratch.add(a);
        return use(a);
    }

    /** True when the plan runs on Metal (simdgroup matrix kernels are available). */
    public boolean isMetal() {
        return Devices.backend() == uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.METAL;
    }

    /** True when the plan runs on CUDA. */
    public boolean isCuda() {
        return Devices.backend() == uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.CUDA;
    }

    public KernelContext context() {
        return new KernelContext();
    }

    public void task(WorkerGrid grid, TaskAdder adder) {
        tasks.add(new Spec("t" + tasks.size(), grid, adder, node));
    }

    /** The vendor-library functions this plan's library tasks call ("cudnn.conv", see {@link Native}). */
    final java.util.Set<String> libraries = new java.util.TreeSet<>();

    /**
     * Adds a TornadoVM library task (a cuBLAS, cuDNN, ... or MLX call, through
     * {@code TaskGraph.libraryTask}): it has no launch grid, and runs on the same device and
     * stream as the kernels around it. Its arrays must come from this emitter's accessors.
     */
    public void library(Native.Lib lib, String function, TaskAdder adder) {
        libraries.add(Native.feature(lib, function));
        Native.emitted(lib);
        tasks.add(new Spec("t" + tasks.size(), null, adder, node));
    }

    /** A 1-D grid of at least {@code n} threads, in groups of {@link #LOCAL}. */
    public static WorkerGrid grid1D(int n) {
        return groups(ceilDiv(Math.max(n, 1), LOCAL));
    }

    /** {@code groups} work-groups of {@link #LOCAL} threads each. */
    public static WorkerGrid groups(int groups) {
        WorkerGrid g = new WorkerGrid1D(Math.max(groups, 1) * LOCAL);
        g.setLocalWork(LOCAL, 1, 1);
        return g;
    }

    /** A 2-D grid covering {@code x * y} threads in tiles of {@code tile * tile}. */
    public static WorkerGrid grid2D(int x, int y, int tile) {
        WorkerGrid g = new WorkerGrid2D(ceilDiv(Math.max(x, 1), tile) * tile, ceilDiv(Math.max(y, 1), tile) * tile);
        g.setLocalWork(tile, tile, 1);
        return g;
    }

    /** {@code groups} work-groups of {@code local} threads. */
    public static WorkerGrid blocks1D(int groups, int local) {
        WorkerGrid g = new WorkerGrid1D(Math.max(groups, 1) * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    /** {@code gx * gy} work-groups of {@code lx * ly} threads. */
    public static WorkerGrid blocks2D(int gx, int gy, int lx, int ly) {
        WorkerGrid g = new WorkerGrid2D(Math.max(gx, 1) * lx, Math.max(gy, 1) * ly);
        g.setLocalWork(lx, ly, 1);
        return g;
    }

    /** {@code gx * gy * gz} work-groups of {@code lx} threads. */
    public static WorkerGrid blocks3D(int gx, int gy, int gz, int lx) {
        WorkerGrid g = new uk.ac.manchester.tornado.api.WorkerGrid3D(Math.max(gx, 1) * lx, Math.max(gy, 1), Math.max(gz, 1));
        g.setLocalWork(lx, 1, 1);
        return g;
    }

    public static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    String graphName() {
        return graphName;
    }

    void registerGrid(GridScheduler scheduler, String graph, Spec s) {
        if (s.grid() == null) {
            return;
        }
        scheduler.addWorkerGrid(graph + "." + s.id(), s.grid());
    }
}
