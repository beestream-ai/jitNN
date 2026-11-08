/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */package ai.beestream.jitnn.engine;

import static ai.beestream.jitnn.engine.TornadoExecutor.*;
import static ai.beestream.jitnn.engine.PlanMemo.*;
import static ai.beestream.jitnn.engine.Telemetry.*;
import static ai.beestream.jitnn.engine.PlanCompiler.*;
import static ai.beestream.jitnn.engine.PlanCache.*;
import static ai.beestream.jitnn.engine.PlanRunner.*;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import ai.beestream.jitnn.Tensor;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * What happens when a plan fails: it is closed, saving what it can, and never run again.
 */
final class PlanRecovery {

    private PlanRecovery() {
    }

    /**
     * Closes a plan whose execution failed: what only it holds (persistent storages, unread
     * outputs) is saved to the host when TornadoVM still can, and no storage refers to it after.
     */
    static void closeFailed(CompiledTrace ct) {
        try {
            close(ct);
        } catch (RuntimeException e) {
            forget(ct);
            closeQuietly(ct);
            for (Storage s : ct.bound) {
                if (s != null) {
                    s.validOn.remove(ct);
                    if (s.boundTrace == ct && !s.hostValid) {
                        s.stale = true;
                        s.boundTrace = null;
                    }
                }
            }
        }
    }

    /** Executes a plan just compiled for a retry; if it fails as well it is closed and the error stands. */
    static void executeOrDrop(CompiledTrace ct, String key, List<Storage> slots) {
        try {
            execute(ct, slots);
        } catch (RuntimeException failure) {
            CACHE.remove(key);
            closeFailed(ct);
            throw failure;
        }
    }

    /**
     * Executes {@code ct}, the plan for {@code layout}; returns the plan that ran, which is a new
     * one when {@code ct} failed and the trace was compiled again: out of device memory (this
     * device's other plans are closed first), or a vendor library failing on its first call (it is
     * turned off and the trace compiled with jitNN's kernels). Any other failure closes the plan and
     * is rethrown. {@code replaced}, the plan {@code ct} replaces (values carried across steps), is
     * closed either way.
     */
    static CompiledTrace executeRecovering(CompiledTrace ct, TraceLayout layout, String slotTag, String nativeTag, CompiledTrace replaced) {
        String k = ct.key;
        String structure = layout.structure();
        List<Storage> slots = layout.slots();
        try {
            try {
                execute(ct, slots);
            } catch (uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException oom) {
                // Device memory is shared by every cached plan. WORKAROUND(TornadoVM): a plan
                // whose execution failed is not safe to execute again, so it is closed (what only
                // it holds saved to the host first), this device's other plans are closed too, and
                // the trace is compiled afresh; with nothing else to free, the error stands.
                CACHE.remove(k);
                closeFailed(ct);
                if (replaced != null) {
                    // its carried values go to the host before the retry, which compiles without them
                    closeFailed(replaced);
                    replaced = null;
                }
                if (!evictAllExcept(null)) {
                    throw oom;
                }
                if (VERBOSE) {
                    System.err.printf("[jitnn] out of device memory in %s: closed this device's cached plans and recompiled%n", ct.name);
                }
                ct = compile(k, layout, null);
                ct.structure = structure;
                BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
                CACHE.put(k, ct);
                compiles++;
                executeOrDrop(ct, k, slots);
            } catch (RuntimeException failure) {
                if (ct.executions > 0 || ct.libraries.isEmpty()) {
                    // a plan whose execution failed is not run again
                    CACHE.remove(k);
                    closeFailed(ct);
                    throw failure;
                }
                // a vendor library failed on its first call (missing at run time, or an argument form it
                // does not take): stop using it and compile the trace with jitNN's kernels
                CACHE.remove(k);
                closeFailed(ct);
                Native.disable(ct.libraries, failure);
                String tag = Native.keyTag();
                structure = Devices.key() + "|" + slotTag + tag + structure.substring(Devices.key().length() + 1 + slotTag.length() + nativeTag.length());
                k = structure + "|P" + layout.ids();
                ct = compile(k, layout, null);
                ct.structure = structure;
                BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
                CACHE.put(k, ct);
                compiles++;
                executeOrDrop(ct, k, slots);
            }
        } finally {
            // the plan this one replaced (carrying values across steps) goes, whether or not this one ran
            if (replaced != null && !replaced.closed) {
                closeFailed(replaced);
            }
        }
        return ct;
    }
}
