package org.beehive.jitnn.engine;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;

/** One trace structure compiled into a TornadoVM execution plan, replayed on every cache hit. */
final class CompiledTrace {

    static final int PERSISTENT = 0;
    static final int INPUT = 1;
    static final int INTERMEDIATE = 2;

    final String key;
    final String name;
    final int[] kinds;
    final Object[] arrays;
    final boolean[] written;
    /** Per slot, the storage the slot is currently bound to (persistent: fixed). */
    final Storage[] bound;
    final int tasks;

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
