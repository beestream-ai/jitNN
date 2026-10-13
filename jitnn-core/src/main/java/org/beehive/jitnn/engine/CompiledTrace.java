package org.beehive.jitnn.engine;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;

/** One trace structure compiled into a TornadoVM execution plan, replayed on every cache hit. */
final class CompiledTrace {

    static final int PERSISTENT = 0;
    static final int INPUT = 1;
    static final int INTERMEDIATE = 2;

    String key;
    /** The key without the identities of the persistent storages (see TornadoExecutor's plan reuse). */
    String structure;
    boolean closed;
    /** The vendor libraries its library tasks call (none for a plan of jitNN kernels only). */
    java.util.Set<String> libraries = java.util.Set.of();
    final String name;
    final int[] kinds;
    final Object[] arrays;
    final boolean[] written;
    /** Per slot, the storage the slot is currently bound to (persistent: fixed). */
    final Storage[] bound;
    final int tasks;

    /** Per slot: an intermediate whose buffer is reused inside the trace (stale after execution). */
    boolean[] recyclable;
    long intermediateBytes;
    /** The arrays some task of the plan reads or writes (identity): only these have device copies. */
    java.util.Set<Object> deviceArrays;
    /** TornadoExecutor's execution count at this plan's latest execution (idle eviction). */
    long lastUsed;

    /** "graph.task" keys of the profiler log and the operator each task belongs to (profiling only). */
    java.util.Map<String, String> taskLabels;
    java.util.Map<String, long[]> profileTotals = new java.util.HashMap<>();

    /** With kernel printing on: the operator listing printed before the kernels on the first run. */
    String printHeader;

    TornadoExecutionPlan plan;
    TornadoExecutionResult last;
    long executions;
    long compileNanos;

    CompiledTrace(String key, String name, int[] kinds, Object[] arrays, boolean[] written, Storage[] bound, int tasks) {
        this.key = key;
        this.name = name;
        this.kinds = kinds;
        this.arrays = arrays;
        this.written = written;
        this.bound = bound;
        this.tasks = tasks;
    }
}
