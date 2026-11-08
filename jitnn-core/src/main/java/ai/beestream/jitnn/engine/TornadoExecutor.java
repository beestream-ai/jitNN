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
 */
package ai.beestream.jitnn.engine;

import static ai.beestream.jitnn.engine.PlanMemo.*;
import static ai.beestream.jitnn.engine.Telemetry.*;
import static ai.beestream.jitnn.engine.PlanCompiler.*;
import static ai.beestream.jitnn.engine.PlanCache.*;
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
 * Compiles a trace into one TornadoVM task graph and runs it.
 *
 * <p>Storages in a trace fall into three kinds. <b>Persistent</b> storages (parameters, optimiser
 * state) are uploaded once and stay on the device; they move between compiled traces only when
 * another trace needs them. <b>Inputs</b> are host data created since the last execution; each
 * execution copies them into the plan's own input arrays and uploads them. <b>Intermediates</b>
 * are produced on the device and fetched only when someone reads them.
 *
 * <p>Compiled plans are cached by the trace's structure (operators, attributes, shapes and which
 * persistent storages it uses), so the second and later training steps replay a plan instead of
 * compiling one. A plan whose persistent storages were all released (their model was closed) is
 * reused by a trace of the same structure over other persistent storages, such as the same model
 * loaded again: their contents are copied into the plan's buffers and uploaded, and nothing is
 * compiled.
 *
 * <p>This class runs a trace from end to end; the parts live in package-private classes:
 * {@link TraceLayout} (slots, kinds, buffer reuse, cache key), {@link PlanMemo} (the replay fast
 * path), {@link PlanCache} (cached plans, reuse, eviction, closing), {@link PlanCompiler} (task
 * graphs), {@link PlanRunner} (binding, staging, execution, fetching), {@link PlanRecovery}
 * (failed plans) and {@link Telemetry} (counters, profiles, timeline).
 */
final class TornadoExecutor {


    static final int CACHE_SIZE = Integer.getInteger("jitnn.plan.cache", 16);
    static final boolean VERBOSE = Boolean.getBoolean("jitnn.verbose");
    /** Host inputs at least this large that are read again become persistent device storage. */
    static final long PROMOTE_BYTES = Long.getLong("jitnn.promoteInputBytes", 16L << 20);
    /** Prints the cache key of every compiled plan (to find what makes a trace recompile). */
    static final boolean DEBUG_KEYS = Boolean.getBoolean("jitnn.debugKeys");
    static final boolean PROFILE = Boolean.getBoolean("jitnn.profile");
    /**
     * Tasks per TornadoVM task graph before the trace is split into a chain of graphs. A graph's
     * high-level bytecode buffer is 8 KiB and a task takes 15 + 5 bytes per argument (jitNN
     * kernels take up to 19), next to the transfer declarations: 40 tasks stays inside it.
     */
    static final int MAX_TASKS = Integer.getInteger("jitnn.graph.maxTasks", 40);
    /** Reuse dead intermediates' buffers within a compiled trace (default on). */
    static final boolean REUSE = Boolean.parseBoolean(System.getProperty("jitnn.reuseBuffers", "true"));
    /**
     * Reuse only buffers of exactly an intermediate's size ({@code -Djitnn.reuseBuffers.exact}): {@code auto}
     * (the default) on Metal with MLX, {@code true} always, {@code false} never.
     */
    static final String EXACT_REUSE = System.getProperty("jitnn.reuseBuffers.exact", "auto");
    /** On CUDA, capture each plan as a CUDA graph and replay it (default on). */
    static final boolean CUDA_GRAPH = Boolean.parseBoolean(System.getProperty("jitnn.cudaGraph", "true"));
    /**
     * Ask TornadoVM to share kernels across array sizes (default on): a new batch or sequence size
     * then reuses the kernels compiled for earlier sizes instead of compiling every one again.
     */
    static final boolean DYNAMIC_SHAPES = Boolean.parseBoolean(System.getProperty("jitnn.dynamicShapes", "true"));
    /**
     * Keep values a plan hands to its own next execution on the device (default on): an input that
     * is an output of the plan's previous execution (a gradient accumulated over steps) is copied
     * across on the device at the end of each execution, instead of being fetched to the host and
     * uploaded again. The plan is compiled once more when such an input is first seen.
     */
    static final boolean CARRY = Boolean.parseBoolean(System.getProperty("jitnn.plan.carry", "true"));
    static volatile boolean loaded;

    private TornadoExecutor() {
    }

    static boolean isLoaded() {
        return loaded;
    }

    /**
     * The executor's lock: plan cache, storage bookkeeping and counters. It is released while
     * TornadoVM runs a plan, so threads in different slots, or on different devices, run their plans
     * at the same time. A plan that writes persistent storages marks them ({@link Storage#writer})
     * first; other threads wait on {@link #WRITTEN} before touching a marked storage.
     */
    static final java.util.concurrent.locks.ReentrantLock LOCK = Storage.LOCK;
    static final java.util.concurrent.locks.Condition WRITTEN = Storage.WRITTEN;

    /**
     * Concurrent executions ({@code -Djitnn.concurrency}, default 1): threads are spread over this
     * many slots, each with its own compiled plans (and device buffers, weights included), so up
     * to this many threads run plans on the device at once. Threads of the same slot and device take
     * turns; threads on different devices (data-parallel ranks) do not wait for each other.
     */
    static final int SLOTS = Math.max(1, Integer.getInteger("jitnn.concurrency", 1));
    static final java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.locks.ReentrantLock> SLOT_LOCKS = new java.util.concurrent.ConcurrentHashMap<>();
    static final java.util.concurrent.atomic.AtomicInteger NEXT_SLOT = new java.util.concurrent.atomic.AtomicInteger();
    static final ThreadLocal<Integer> SLOT = ThreadLocal.withInitial(() -> NEXT_SLOT.getAndIncrement() % SLOTS);

    static void run(List<Node> recorded, Tensor[] wanted) {
        int slot = SLOT.get();
        java.util.concurrent.locks.ReentrantLock slotLock = SLOT_LOCKS.computeIfAbsent(Devices.key() + "#" + slot, k -> new java.util.concurrent.locks.ReentrantLock());
        slotLock.lock();
        try {
            LOCK.lock();
            try {
                runLocked(recorded, wanted, slot);
            } finally {
                // the signature's storages are not kept past the trace (weights of a closed model, inputs)
                Signed sg = MEMO ? SIGNED.get() : null;
                if (sg != null) {
                    Arrays.fill(sg.locals, null);
                }
                LOCK.unlock();
            }
        } finally {
            slotLock.unlock();
        }
        if (Autotune.enabled()) {
            // choices met in this plan's compile are measured now, with no lock held
            Autotune.tunePending();
        }
    }

    static void runLocked(List<Node> recorded, Tensor[] wanted, int slot) {
        long start = System.nanoTime();
        long before = executeNanos;
        loaded = true;
        String slotTag = SLOTS > 1 ? "S" + slot + "|" : "";
        String nativeTag = Native.keyTag();
        String global = null;
        int hash = 0;
        Signed sg = MEMO ? SIGNED.get() : null;
        if (MEMO && signature(sg, recorded, wanted)) {
            global = Devices.key() + '|' + slotTag + nativeTag + Autotune.keyTag();
            hash = memoHash(sg);
            if (replay(sg, hash, global, recorded, wanted)) {
                hostNanos += System.nanoTime() - start - (executeNanos - before);
                return;
            }
        }
        List<Node> nodes = ai.beestream.jitnn.ops.Fusion.rewrite(recorded, wanted);
        String prefix = Devices.key() + '|' + slotTag + nativeTag + Autotune.keyTag();
        TraceLayout layout = TraceLayout.of(nodes, wanted, prefix);
        List<Storage> slots = layout.slots();
        boolean[] written = layout.written();
        String structure = layout.structure();
        String k = structure + "|P" + layout.ids();
        CompiledTrace ct = CACHE.get(k);
        // remembered only once the rewrite has given the same plan twice: the first calls of a model
        // fold weights (batch norm, channels-last copies) that later calls take from the caches
        boolean stable = ct != null;
        if (ct == null && (ct = reuse(structure, slots, layout.kinds())) != null) {
            CACHE.remove(ct.key);
            ct.key = k;
            CACHE.put(k, ct);
            reuses++;
        } else if (ct == null) {
            if (DEBUG_KEYS) {
                System.err.println("[jitnn] key " + k);
            }
            ct = compile(k, layout, null);
            ct.structure = structure;
            BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
            CACHE.put(k, ct);
            compiles++;
        } else {
            replays++;
        }
        CompiledTrace replaced = null;
        if (ct.seenCarry != null && ct.carry == null) {
            // inputs of this plan were outputs of its previous execution (gradients accumulated over
            // steps): compile it once more to copy them across on the device; the old plan, which
            // still holds them, is closed once the new one has read them
            replaced = ct;
            stable = false;
            ct = compile(k, layout, replaced.seenCarry.clone());
            if (VERBOSE) {
                System.err.printf("[jitnn] %s replaces %s: %d input(s) carried on the device from the previous execution%n", ct.name, replaced.name,
                        Arrays.stream(ct.carry).filter(j -> j >= 0).count());
            }
            ct.structure = structure;
            BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
            CACHE.put(k, ct);
            compiles++;
        }
        ct.lastUsed = ++executionsTotal;
        lastIntermediateBytes = ct.intermediateBytes;
        evictIdle(ct);
        CompiledTrace ran = executeRecovering(ct, layout, slotTag, nativeTag, replaced);
        if (ran != ct) {
            stable = false;
            ct = ran;
        }
        for (Node n : nodes) {
            n.executed = true;
        }
        for (Node n : recorded) {
            n.executed = true;
        }
        for (int i = 0; i < slots.size(); i++) {
            if (written[i]) {
                slots.get(i).version++;
            }
        }
        for (Tensor t : wanted) {
            Storage s = t.storage();
            if (!s.hostValid && layout.slotOf().containsKey(s)) {
                fetch(s);
            }
        }
        if (global != null && stable) {
            remember(sg, hash, global, k, ct, slots, written, recorded.size());
        }
        hostNanos += System.nanoTime() - start - (executeNanos - before);
    }

    static int add(List<Storage> slots, Storage s) {
        slots.add(s);
        return slots.size() - 1;
    }
}
