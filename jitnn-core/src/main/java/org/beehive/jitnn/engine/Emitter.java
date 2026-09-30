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

    record Spec(String id, WorkerGrid grid, TaskAdder adder) {
    }

    private final String graphName;
    private final Map<Storage, Object> arrays;
    final List<Spec> tasks = new ArrayList<>();
    final List<Object> constants = new ArrayList<>();
    private final Map<String, IntArray> constantCache = new java.util.HashMap<>();

    Emitter(String graphName, IdentityHashMap<Storage, Object> arrays) {
        this.graphName = graphName;
        this.arrays = arrays;
    }

    public FloatArray f32(Tensor t) {
        return (FloatArray) arrays.get(t.storage());
    }

    public IntArray i32(Tensor t) {
        return (IntArray) arrays.get(t.storage());
    }

    /** A constant int array uploaded once with the graph (shape and stride metadata). */
    public IntArray constant(int... values) {
        String key = java.util.Arrays.toString(values);
        return constantCache.computeIfAbsent(key, k -> {
            IntArray a = IntArray.fromArray(values.length == 0 ? new int[] { 0 } : values);
            constants.add(a);
            return a;
        });
    }

    public KernelContext context() {
        return new KernelContext();
    }

    public void task(WorkerGrid grid, TaskAdder adder) {
        tasks.add(new Spec("t" + tasks.size(), grid, adder));
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

    public static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    String graphName() {
        return graphName;
    }

    void registerGrids(GridScheduler scheduler) {
        for (Spec s : tasks) {
            scheduler.addWorkerGrid(graphName + "." + s.id(), s.grid());
        }
    }
}
