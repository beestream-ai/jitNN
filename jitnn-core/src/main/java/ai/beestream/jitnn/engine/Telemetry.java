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
 * The executor's counters, profiles and timeline (Chrome trace events).
 */
final class Telemetry {

    private Telemetry() {
    }

    static long hostNanos;
    static long executeNanos;
    static long compiles;
    static long replays;
    static long reuses;

    /** The sum of every {@code "field": "n"} in TornadoVM's profiler log (nanoseconds). */
    static long sumField(String log, String field) {
        long sum = 0;
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("\"" + field + "\"\\s*:\\s*\"?(\\d+)").matcher(log == null ? "" : log);
        while (m.find()) {
            sum += Long.parseLong(m.group(1));
        }
        return sum;
    }

    static final java.util.regex.Pattern TASK_TIME = java.util.regex.Pattern.compile(
            "\"([A-Za-z0-9_]+\\.t\\d+)\"\\s*:\\s*\\{[^}]*?\"TASK_KERNEL_TIME\"\\s*:\\s*\"(\\d+)\"");
    static final java.util.Map<String, long[]> BREAKDOWN = new java.util.TreeMap<>();

    /** Adds the per-task kernel times of one profiled execution to the per-operator totals. */
    static void accumulate(CompiledTrace ct, String log) {
        if (ct.taskLabels == null || log == null) {
            return;
        }
        // TornadoVM's log keeps every execution's task entries: total it, and add the difference
        // from the plan's previous totals.
        java.util.Map<String, long[]> now = new java.util.HashMap<>();
        java.util.regex.Matcher m = TASK_TIME.matcher(log);
        while (m.find()) {
            String label = ct.taskLabels.get(m.group(1));
            if (label != null) {
                long[] acc = now.computeIfAbsent(label, k -> new long[2]);
                acc[0]++;
                acc[1] += Long.parseLong(m.group(2));
            }
        }
        for (var en : now.entrySet()) {
            long[] before = ct.profileTotals.getOrDefault(en.getKey(), new long[2]);
            long[] acc = BREAKDOWN.computeIfAbsent(en.getKey(), k -> new long[2]);
            acc[0] += en.getValue()[0] - before[0];
            acc[1] += en.getValue()[1] - before[1];
        }
        ct.profileTotals = now;
    }

    // ---------------------------------------------------------------- timeline (Chrome trace events)

    static final java.util.regex.Pattern TASK_ID = java.util.regex.Pattern.compile("(.*?)(?:_(\\d+))?\\.t(\\d+)");
    static List<String> TRACE;
    static long traceOrigin;

    static void traceStart() {
        LOCK.lock();
        try {
            TRACE = java.util.Collections.synchronizedList(new ArrayList<>());
            traceOrigin = System.nanoTime();
        } finally {
            LOCK.unlock();
        }
    }

    /** The recorded events, and recording stops. */
    static List<String> traceStop() {
        LOCK.lock();
        try {
            List<String> events = TRACE == null ? List.of() : new ArrayList<>(TRACE);
            TRACE = null;
            return events;
        } finally {
            LOCK.unlock();
        }
    }

    static boolean tracing() {
        return TRACE != null;
    }

    static long traceOrigin() {
        return traceOrigin;
    }

    /** Adds an event (a JSON object) to the timeline, when recording. */
    static void traceEvent(String json) {
        LOCK.lock();
        try {
            if (TRACE != null) {
                TRACE.add(json);
            }
        } finally {
            LOCK.unlock();
        }
    }

    /**
     * One plan execution on the timeline: its host span (thread 1), and its kernels' device times
     * from the profiler (thread 2), in launch order and laid back to back from the execution's start
     * (TornadoVM reports durations, not start times).
     */
    static void traceExecution(CompiledTrace ct, String log, long start, long end) {
        java.util.Map<String, long[]> now = new java.util.HashMap<>();
        java.util.regex.Matcher m = TASK_TIME.matcher(log == null ? "" : log);
        while (m.find()) {
            long[] acc = now.computeIfAbsent(m.group(1), k -> new long[2]);
            acc[0]++;
            acc[1] += Long.parseLong(m.group(2));
        }
        List<String> tasks = new ArrayList<>(now.keySet());
        tasks.sort(java.util.Comparator.comparingLong(Telemetry::taskOrder));
        String first = ct.executions == 1 ? " (first run: compiles the kernels)" : "";
        TRACE.add(String.format(java.util.Locale.ROOT, "{\"name\":\"%s%s\",\"cat\":\"plan\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":1}", ct.name, first,
                (start - traceOrigin) / 1e3, (end - start) / 1e3));
        double at = (start - traceOrigin) / 1e3;
        for (String task : tasks) {
            long[] before = ct.taskTotals.getOrDefault(task, new long[2]);
            long ns = now.get(task)[1] - before[1];
            if (now.get(task)[0] == before[0]) {
                continue;
            }
            String label = ct.taskLabels != null ? ct.taskLabels.getOrDefault(task, task) : task;
            TRACE.add(String.format(java.util.Locale.ROOT, "{\"name\":\"%s\",\"cat\":\"kernel\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":2,\"args\":{\"plan\":\"%s\",\"task\":\"%s\"}}",
                    label.replace("\"", "'"), at, ns / 1e3, ct.name, task));
            at += ns / 1e3;
        }
        ct.taskTotals = now;
    }

    /** Graph index then task index of a task id "plan[_g].tN", for launch order. */
    static long taskOrder(String task) {
        java.util.regex.Matcher m = TASK_ID.matcher(task);
        if (!m.matches()) {
            return Long.MAX_VALUE;
        }
        long graph = m.group(2) == null ? 0 : Long.parseLong(m.group(2));
        return graph * 1_000_000L + Long.parseLong(m.group(3));
    }

    /** With -Djitnn.profile=true: per operator task ("op#k"), [launches, device ns] since the last call; resets. */
    static java.util.Map<String, long[]> kernelBreakdown() {
        LOCK.lock();
        try {
            java.util.Map<String, long[]> copy = new java.util.TreeMap<>(BREAKDOWN);
            BREAKDOWN.clear();
            return copy;
        } finally {
            LOCK.unlock();
        }
    }

    /**
     * This thread's last plan execution: {@code [device kernel ns, TornadoVM total ns]} (with
     * -Djitnn.profile=true) and {@code [2]} its intermediates' device bytes. Per thread, so that
     * threads on other devices or slots do not overwrite each other's.
     */
    static final ThreadLocal<long[]> LAST = ThreadLocal.withInitial(() -> new long[3]);

    /** With -Djitnn.profile=true: [device kernel ns, TornadoVM total ns] of this thread's last execution. */
    static long[] lastProfile() {
        long[] l = LAST.get();
        return new long[] { l[0], l[1] };
    }

    /**
     * Host-device traffic the executor causes: [persistent or output fetches to the host,
     * their elements, persistent re-uploads (a storage written by another plan), their elements,
     * input copies, their elements].
     */
    static final long[] transfers = new long[6];

    /**
     * Bytes an operator reads and writes at least: its input and output tensors, once each (a
     * fused operator no longer writes and reads back the intermediates it replaced).
     */
    static long tensorBytes(Node n) {
        long b = 0;
        for (Tensor t : n.inputs) {
            b += t == null ? 0 : bytes(t);
        }
        for (Tensor t : n.outputs) {
            b += bytes(t);
        }
        return b;
    }

    static long bytes(Tensor t) {
        ai.beestream.jitnn.DType d = t.dtype();
        long numel = t.numel();
        return d.isBlockQuantized() ? numel / d.blockValues() * d.blockBytes() : numel * d.bytes();
    }

    /** Per plan execution so far: {@code [executions, kernels, library calls, tensor bytes]}. */
    static final long[] TRAFFIC = new long[4];

    static long[] trafficStats() {
        LOCK.lock();
        try {
            return TRAFFIC.clone();
        } finally {
            LOCK.unlock();
        }
    }

    static long[] transferStats() {
        LOCK.lock();
        try {
            return transfers.clone();
        } finally {
            LOCK.unlock();
        }
    }

    /** Cumulative [jitNN host ns outside plan.execute, ns inside plan.execute] of run(). */
    static long[] timeSplit() {
        LOCK.lock();
        try {
            return new long[] { hostNanos, executeNanos };
        } finally {
            LOCK.unlock();
        }
    }

    /** Device bytes of intermediates (buffers other than inputs and persistent storages) of this thread's last plan run. */
    static long lastIntermediateBytes() {
        return LAST.get()[2];
    }

    /** Compile and replay counts, for tests and benchmarks. */
    static long[] stats() {
        LOCK.lock();
        try {
            return new long[] { compiles, replays, reuses, memoReplays };
        } finally {
            LOCK.unlock();
        }
    }
}
