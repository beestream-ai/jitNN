package ai.beestream.jitnn.engine;

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
    /** Being run by a thread that released the executor's lock: not closed, evicted or reused meanwhile. */
    volatile boolean busy;
    /** The vendor libraries its library tasks call (none for a plan of jitNN kernels only). */
    java.util.Set<String> libraries = java.util.Set.of();
    /** Kernel tasks, library-call tasks, and the bytes its operators read and write at least ({@link TornadoExecutor#tensorBytes}). */
    int kernels;
    int libraryCalls;
    long tensorBytes;
    /** The autotuned choices its operators asked for (closed once they are measured). */
    java.util.Set<String> choices = java.util.Set.of();
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
    /**
     * Per input slot, the intermediate slot whose value the plan copies into it on the device at the
     * end of each execution, for the next one to read (-1: the input comes from the host); null when
     * the plan was compiled without carried values.
     */
    int[] carry;
    /** Per input slot, the intermediate slot of the previous execution it was seen to read (-1: none); null when none was seen. */
    int[] seenCarry;
    /**
     * Per input slot, the plan whose device buffer it reads (an output of that plan's latest
     * execution, consumed on the device), or null when it comes from the host; null when none does.
     */
    CompiledTrace[] importFrom;
    /** Per imported input slot, the producing plan's slot. */
    int[] importSlot;
    /** Per persistent slot, the plan whose device buffer of that storage this plan uses (null: its own). */
    CompiledTrace[] sharedFrom;
    /** Whether {@link #close} has run (forget only takes the plan out of the indexes). */
    boolean released;
    /** The task graph of this plan that holds each of its arrays on the device (for plans importing them). */
    java.util.IdentityHashMap<Object, uk.ac.manchester.tornado.api.ImmutableTaskGraph> graphOf;
    /** Plans that import this plan's buffers: closed with it. */
    java.util.List<CompiledTrace> dependents = new java.util.ArrayList<>();

    /** "graph.task" keys of the profiler log and the operator each task belongs to (profiling only). */
    java.util.Map<String, String> taskLabels;
    java.util.Map<String, long[]> profileTotals = new java.util.HashMap<>();
    /** Per task id: [launches, device ns] in the profiler log so far (for the timeline). */
    java.util.Map<String, long[]> taskTotals = new java.util.HashMap<>();

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
