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
import static ai.beestream.jitnn.engine.PlanRecovery.*;
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
import ai.beestream.jitnn.JitnnConfig;
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
 * The compiled plans: the cache keyed by trace structure, reuse of a plan for new persistent
 * storages, eviction of idle and least recently used plans, and closing a plan (what only it holds
 * is saved to the host first).
 */
final class PlanCache {

    private PlanCache() {
    }

    static final Map<String, CompiledTrace> CACHE = new LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, CompiledTrace> eldest) {
            // every concurrency slot compiles its own plans, so the cache holds CACHE_SIZE per slot
            // and device: the least recently used plan of this device that no thread is running is
            // closed (another device's plans are closed by its own threads)
            int limit = Math.max(CACHE_SIZE, RESERVED.get()) * SLOTS;
            int local = 0;
            for (CompiledTrace c : values()) {
                if (local(c)) {
                    local++;
                }
            }
            if (local <= limit) {
                return false;
            }
            // never the plan just added (the last entry), which its caller is about to run
            CompiledTrace victim = null;
            CompiledTrace newest = null;
            for (CompiledTrace c : values()) {
                if (victim == null && local(c) && !c.busy) {
                    victim = c;
                }
                newest = c;
            }
            if (victim != null && victim != newest) {
                values().remove(victim);
                close(victim);
            }
            return false;
        }
    };

    /**
     * Whether {@code ct} runs on this thread's device. Plans are closed only by threads of their own
     * device: another device's thread may be launching one of that device's plans meanwhile (the
     * ranks of data-parallel training run their plans at the same time).
     */
    static boolean local(CompiledTrace ct) {
        return ct.key.startsWith(Devices.key() + "|");
    }

    /** Plan cache room reserved by users that need many plans at once (Engine.reservePlans: shape buckets). */
    static final java.util.concurrent.atomic.AtomicInteger RESERVED = new java.util.concurrent.atomic.AtomicInteger();

    /** Cached plans by structure (the key without persistent storage identities), for plan reuse. */
    static final Map<String, List<CompiledTrace>> BY_STRUCTURE = new java.util.HashMap<>();

    /**
     * Closes this device's plans that used any of the measured {@code choices} (their next run
     * compiles with the decision) and the measurement plans (those with forced variants).
     */
    static void forgetTuning(java.util.Set<String> choices) {
        LOCK.lock();
        try {
            var it = CACHE.values().iterator();
            while (it.hasNext()) {
                CompiledTrace ct = it.next();
                boolean measured = ct.key.contains("AT{");
                boolean stale = !java.util.Collections.disjoint(ct.choices, choices);
                if ((measured || stale) && !ct.busy && local(ct)) {
                    close(ct);
                    it.remove();
                }
            }
        } finally {
            LOCK.unlock();
        }
    }

    /**
     * A cached plan of this structure whose persistent storages are those of the trace or were
     * released, rebound to the trace's storages; null if there is none. Each rebound storage takes
     * over the plan's buffer (its host contents are copied in, and uploaded at the next execution).
     */
    static CompiledTrace reuse(String structure, List<Storage> slots, int[] kinds) {
        List<CompiledTrace> candidates = BY_STRUCTURE.get(structure);
        if (candidates == null) {
            return null;
        }
        for (CompiledTrace c : candidates) {
            boolean fits = !c.closed && !c.busy;
            for (int i = 0; fits && i < kinds.length; i++) {
                Storage s = slots.get(i);
                Storage old = c.bound[i];
                if (kinds[i] != CompiledTrace.PERSISTENT || old == s) {
                    continue;
                }
                // the storage's array may be another plan's buffer: it can move only if no plan holds it
                boolean movable = s.array == c.arrays[i] || (!s.planBound && s.validOn.isEmpty());
                fits = old != null && old.released && movable;
            }
            if (!fits) {
                continue;
            }
            int rebound = 0;
            for (int i = 0; i < kinds.length; i++) {
                Storage s = slots.get(i);
                Storage old = c.bound[i];
                if (kinds[i] != CompiledTrace.PERSISTENT || old == s) {
                    continue;
                }
                TornadoNativeArray buffer = (TornadoNativeArray) c.arrays[i];
                old.validOn.remove(c);
                if (old.array == buffer) {
                    old.array = null;
                    old.hostValid = false;
                }
                if (s.array != buffer) {
                    Engine.ensureHost(s);
                    copy(s.array, buffer);
                    s.array = buffer;
                    s.hostValid = true;
                }
                s.planBound = true;
                c.bound[i] = s;
                rebound++;
            }
            if (VERBOSE) {
                System.err.printf("[jitnn] reused plan %s for released persistent storages: %d rebound, nothing compiled%n", c.name, rebound);
            }
            return c;
        }
        return null;
    }

    /** Releases a plan that failed before it ran (nothing of it is on the host to save). */
    static void closeQuietly(CompiledTrace ct) {
        try {
            ct.plan.close();
        } catch (TornadoExecutionPlanException | RuntimeException ignored) {
            // the plan is unusable either way
        }
    }

    /**
     * Drops a plan from the structure index and from the replay memos (it was closed or dropped from
     * the cache). A memo would otherwise keep the closed plan, its host arrays and the storages it was
     * bound to reachable until 64 more traces were remembered.
     */
    static void forget(CompiledTrace ct) {
        ct.closed = true;
        List<CompiledTrace> list = ct.structure == null ? null : BY_STRUCTURE.get(ct.structure);
        if (list != null) {
            list.remove(ct);
            if (list.isEmpty()) {
                BY_STRUCTURE.remove(ct.structure);
            }
        }
        var memos = MEMOS.values().iterator();
        while (memos.hasNext()) {
            List<Memo> l = memos.next();
            int before = l.size();
            l.removeIf(m -> m.ct() == ct);
            memoCount -= before - l.size();
            if (l.isEmpty()) {
                memos.remove();
            }
        }
    }

    static void close(CompiledTrace ct) {
        forget(ct);
        for (int i = 0; i < ct.kinds.length; i++) {
            Storage s = ct.bound[i];
            if (s == null) {
                continue;
            }
            if (ct.kinds[i] == CompiledTrace.PERSISTENT) {
                if (s.validOn.contains(ct) && !s.hostValid && s.validOn.size() == 1 && !s.released) {
                    ct.last.transferToHost(s.array);
                    s.hostValid = true;
                }
                s.validOn.remove(ct);
            } else if (s.boundTrace == ct && !s.hostValid && !s.stale && ct.last != null && ct.deviceArrays.contains(s.array)) {
                // (an array no task touched has no device copy; asking TornadoVM for it corrupts the plan's state)
                // an output not read yet: keep its data on the host, as the plan's buffers go. The
                // plan's host array becomes the storage's own, unless another slot of the plan shares it
                ct.last.transferToHost(s.array);
                if (sharedArray(ct, s.array)) {
                    TornadoNativeArray own = Storage.newArray(s.dtype, s.numel);
                    copy((TornadoNativeArray) s.array, own);
                    s.array = own;
                }
                s.hostValid = true;
                s.boundTrace = null;
            }
        }
        try {
            ct.plan.close();
        } catch (TornadoExecutionPlanException e) {
            throw new ai.beestream.jitnn.JitnnException("closing plan " + ct.name + " failed", e);
        }
    }

    /** Whether more than one slot of {@code ct} uses {@code array} (a recycled buffer). */
    static boolean sharedArray(CompiledTrace ct, Object array) {
        int uses = 0;
        for (Object a : ct.arrays) {
            if (a == array && ++uses > 1) {
                return true;
            }
        }
        return false;
    }

    /** Executions of any plan so far; a plan's lastUsed is this count at its latest execution. */
    static long executionsTotal;

    /**
     * Plans holding at least this many bytes of intermediates are closed once IDLE_EXECUTIONS
     * executions of other plans pass without them (a first step that is never repeated, a
     * validation pass between epochs), freeing their device buffers. A closed plan that comes back
     * is compiled again.
     */
    static final long IDLE_BYTES = JitnnConfig.current().get(JitnnConfig.PLAN_IDLE_BYTES);
    static final long IDLE_EXECUTIONS = JitnnConfig.current().get(JitnnConfig.PLAN_IDLE_EXECUTIONS);

    static void evictIdle(CompiledTrace running) {
        if ((executionsTotal & 31) != 0) {
            return;
        }
        var it = CACHE.values().iterator();
        while (it.hasNext()) {
            CompiledTrace ct = it.next();
            if (ct != running && !ct.busy && local(ct) && (ct.intermediateBytes >= IDLE_BYTES || orphaned(ct)) && executionsTotal - ct.lastUsed > IDLE_EXECUTIONS) {
                if (VERBOSE) {
                    System.err.printf("[jitnn] closed idle plan %s (%.1f MB of intermediates)%n", ct.name, ct.intermediateBytes / 1e6);
                }
                close(ct);
                it.remove();
            }
        }
    }

    /**
     * True when every weight the plan binds was released (its model was closed): the plan stays
     * for a model of the same structure, and is closed once idle.
     */
    static boolean orphaned(CompiledTrace ct) {
        boolean any = false;
        for (Storage s : ct.bound) {
            if (s != null && s.persistent) {
                if (!s.released) {
                    return false;
                }
                any = true;
            }
        }
        return any;
    }

    /** Closes every cached plan except {@code keep}; false when there was nothing to close. */
    static boolean evictAllExcept(CompiledTrace keep) {
        boolean any = false;
        var it = CACHE.values().iterator();
        while (it.hasNext()) {
            CompiledTrace ct = it.next();
            if (ct != keep && !ct.busy && local(ct)) {
                close(ct);
                it.remove();
                any = true;
            }
        }
        return any;
    }

    static void closeAll() {
        LOCK.lock();
        try {
            // a plan another thread is running is closed once it has finished
            while (CACHE.values().stream().anyMatch(c -> c.busy)) {
                WRITTEN.awaitUninterruptibly();
            }
            for (CompiledTrace ct : CACHE.values()) {
                close(ct);
            }
            CACHE.clear();
            MEMOS.clear();
            memoCount = 0;
        } finally {
            LOCK.unlock();
        }
    }
}
