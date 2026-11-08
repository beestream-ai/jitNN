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
import static ai.beestream.jitnn.engine.PlanRecovery.*;

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
 * Runs a compiled plan: binds the trace's storages to the plan's buffers, stages inputs, executes
 * with the executor's lock released, and fetches outputs on demand.
 */
final class PlanRunner {

    private PlanRunner() {
    }

    /** Waits (releasing {@link #LOCK}) until no other thread's plan is writing {@code s}. Called with LOCK held. */
    static void awaitWritten(Storage s, CompiledTrace self) {
        while (s.writer != null && s.writer != self) {
            WRITTEN.awaitUninterruptibly();
        }
    }

    static final Object PRINT_LOCK = new Object();

    /** The first run of a plan with kernel printing on: collects TornadoVM's source dump from standard output. */
    static TornadoExecutionResult executePrinting(CompiledTrace ct) {
        java.io.PrintStream sink = Engine.kernelSink();
        synchronized (PRINT_LOCK) {
            java.io.PrintStream stdout = System.out;
            java.io.ByteArrayOutputStream dump = new java.io.ByteArrayOutputStream();
            System.setOut(new java.io.PrintStream(dump, true));
            TornadoExecutionResult result;
            try {
                result = ct.plan.execute();
            } finally {
                System.setOut(stdout);
                ct.plan.withoutPrintKernel();
                String header = ct.printHeader;
                ct.printHeader = null;
                if (sink != null) {
                    sink.print(header);
                    sink.print(dump.toString());
                    sink.println("// ==== end of plan " + ct.name);
                    sink.flush();
                }
            }
            return result;
        }
    }

    /** The plan this thread executed last (Autotune times it again). */
    static final ThreadLocal<java.lang.ref.WeakReference<CompiledTrace>> LAST = new ThreadLocal<>();

    /**
     * Runs the plan this thread executed last {@code runs} more times, as it is (no recording, no
     * lookup), and returns the median seconds per execution: the device time of its kernels plus
     * the plan's launch. For Autotune, on its measurement plans.
     */
    static double timeLast(int runs) {
        java.lang.ref.WeakReference<CompiledTrace> ref = LAST.get();
        CompiledTrace ct = ref == null ? null : ref.get();
        if (ct == null || ct.closed) {
            throw new IllegalStateException("no plan to time: this thread's last plan was closed");
        }
        // marked as running, so no other thread closes it meanwhile
        LOCK.lock();
        try {
            ct.busy = true;
        } finally {
            LOCK.unlock();
        }
        double[] t = new double[runs];
        try {
            for (int r = -2; r < runs; r++) {
                long t0 = System.nanoTime();
                ct.plan.execute();
                if (r >= 0) {
                    t[r] = (System.nanoTime() - t0) / 1e9;
                }
            }
        } finally {
            LOCK.lock();
            try {
                ct.busy = false;
                WRITTEN.signalAll();
            } finally {
                LOCK.unlock();
            }
        }
        java.util.Arrays.sort(t);
        return t[runs / 2];
    }

    static void execute(CompiledTrace ct, List<Storage> slots) {
        LAST.set(new java.lang.ref.WeakReference<>(ct));
        // persistent storages another thread's plan is writing are read or written only after it
        for (int i = 0; i < slots.size(); i++) {
            if (ct.kinds[i] == CompiledTrace.PERSISTENT) {
                awaitWritten(slots.get(i), ct);
            }
        }
        // Inputs first: an input may be an intermediate of this plan's previous execution, which
        // must be fetched before its slot is rebound below (unless the plan carries it on the device).
        for (int i = 0; i < slots.size(); i++) {
            if (ct.kinds[i] == CompiledTrace.INPUT) {
                Storage s = slots.get(i);
                int from = CARRY ? previousOutput(ct, s) : -1;
                boolean carried = ct.carry != null && ct.carry[i] >= 0;
                if (carried && from == ct.carry[i]) {
                    // the end of the previous execution copied it into this input's device buffer
                    continue;
                }
                if (from >= 0 && ct.carry == null) {
                    if (ct.seenCarry == null) {
                        ct.seenCarry = new int[slots.size()];
                        Arrays.fill(ct.seenCarry, -1);
                    }
                    ct.seenCarry[i] = from;
                }
                Engine.ensureHost(s);
                transfers[4]++;
                transfers[5] += s.numel;
                TornadoNativeArray to = (TornadoNativeArray) ct.arrays[i];
                if (s.array == null && s.heap != null) {
                    // a shared Java array (Tensor.wrap): one copy, straight into the plan's buffer
                    MemorySegment heap = s.heap instanceof float[] f ? MemorySegment.ofArray(f) : MemorySegment.ofArray((int[]) s.heap);
                    stage(heap, to.getSegment(), (long) s.numel * 4);
                } else if (s.array != to) {
                    TornadoNativeArray own = (TornadoNativeArray) s.array;
                    stage(own.getSegment(), to.getSegment(), Math.min(own.getNumBytesOfSegment(), to.getNumBytesOfSegment()));
                }
                if (carried && ct.executions > 0) {
                    // a carried input is uploaded on the first execution only: another value is sent now
                    ct.plan.transferToDevice(ct.arrays[i]);
                }
            }
        }
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (ct.kinds[i] == CompiledTrace.PERSISTENT && !s.validOn.contains(ct)) {
                if (!s.hostValid) {
                    fetch(s);
                }
                if (ct.executions > 0) {
                    transfers[2]++;
                    transfers[3] += s.numel;
                    ct.plan.transferToDevice(s.array);
                }
            }
        }
        for (int i = 0; i < slots.size(); i++) {
            if (ct.kinds[i] != CompiledTrace.INTERMEDIATE) {
                continue;
            }
            Storage s = slots.get(i);
            Storage previous = ct.bound[i];
            if (previous != null && previous != s) {
                if (previous.dropped) {
                    // released by the user: nothing to keep
                } else if (previous.hostValid) {
                    // Keep the host copy the user may still read: detach it from the plan's array,
                    // into a pooled array that returns when the tensor is released or collected.
                    TornadoNativeArray own = HostPool.take(previous.dtype, previous.numel);
                    copy((TornadoNativeArray) previous.array, own);
                    previous.array = own;
                    previous.lease = HostPool.lease(previous, own);
                } else {
                    previous.stale = true;
                }
                previous.boundTrace = null;
            }
            s.array = (TornadoNativeArray) ct.arrays[i];
            ct.bound[i] = s;
        }
        long t0 = System.nanoTime();
        boolean writes = false;
        for (boolean w : ct.written) {
            writes |= w;
        }
        long[] readVersions = null;
        if (ct.printHeader != null) {
            ct.last = executePrinting(ct);
        } else {
            // The persistent storages this plan writes are marked: other threads wait for them
            // (awaitWritten) instead of for the lock, which is released while TornadoVM runs the plan,
            // so plans on other devices (data-parallel ranks, whose collectives wait for each
            // other) run at the same time.
            ct.busy = true;
            if (writes) {
                for (int i = 0; i < slots.size(); i++) {
                    if (ct.written[i]) {
                        slots.get(i).writer = ct;
                    }
                }
            }
            // versions of the persistent storages this plan only reads: another thread's plan may
            // write one while the lock is released, and then this plan's copy is not the newest
            readVersions = new long[slots.size()];
            for (int i = 0; i < slots.size(); i++) {
                if (ct.kinds[i] == CompiledTrace.PERSISTENT && !ct.written[i]) {
                    readVersions[i] = slots.get(i).version;
                }
            }
            int holds = LOCK.getHoldCount();
            for (int h = 0; h < holds; h++) {
                LOCK.unlock();
            }
            try {
                ct.last = ct.plan.execute();
            } finally {
                for (int h = 0; h < holds; h++) {
                    LOCK.lock();
                }
                ct.busy = false;
                if (writes) {
                    for (int i = 0; i < slots.size(); i++) {
                        if (slots.get(i).writer == ct) {
                            slots.get(i).writer = null;
                        }
                    }
                }
                // threads waiting for a plan to finish: writers, fetch() and closeAll()
                WRITTEN.signalAll();
            }
        }
        executeNanos += System.nanoTime() - t0;
        if (VERBOSE && ct.executions == 0) {
            String split = "";
            if (PROFILE) {
                // TornadoVM's per-task compile times: the compiler (Java to CUDA C, OpenCL C, MSL) and the driver (to binary)
                String log = ct.last.getProfilerResult().getProfileLog();
                // Java to kernel source: Graal's time on older TornadoVM, the TC compiler's on newer ones
                double front = (sumField(log, "TASK_COMPILE_GRAAL_TIME") + sumField(log, "TASK_COMPILE_TC_TIME")) / 1e6;
                split = String.format(" (compiler %.1f ms, driver %.1f ms)", front, sumField(log, "TASK_COMPILE_DRIVER_TIME") / 1e6);
            }
            System.err.printf("[jitnn] first execution of %s (kernel compilation): %.1f ms%s%n", ct.name, (System.nanoTime() - t0) / 1e6, split);
        }
        if (ct.executions == 0 && !ct.libraries.isEmpty()) {
            Native.proven(ct.libraries);
        }
        ct.executions++;
        TRAFFIC[0]++;
        TRAFFIC[1] += ct.kernels;
        TRAFFIC[2] += ct.libraryCalls;
        TRAFFIC[3] += ct.tensorBytes;
        if (PROFILE) {
            Telemetry.LAST.get()[0] = ct.last.getProfilerResult().getDeviceKernelTime();
            Telemetry.LAST.get()[1] = ct.last.getProfilerResult().getTotalTime();
            accumulate(ct, ct.last.getProfilerResult().getProfileLog());
            if (TRACE != null) {
                traceExecution(ct, ct.last.getProfilerResult().getProfileLog(), t0, System.nanoTime());
            }
        }
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (ct.kinds[i] == CompiledTrace.PERSISTENT) {
                if (ct.written[i]) {
                    s.validOn.clear();
                    s.hostValid = false;
                    s.validOn.add(ct);
                } else if (readVersions == null || s.version == readVersions[i]) {
                    s.validOn.add(ct);
                }
            } else if (ct.kinds[i] == CompiledTrace.INTERMEDIATE) {
                s.boundTrace = ct;
                s.boundExec = ct.executions;
                s.hostValid = false;
                // A recycled buffer holds a later tensor's data now.
                s.stale = ct.recyclable != null && ct.recyclable[i];
            }
        }
    }

    static CompiledTrace idleCopy(Storage s) {
        for (CompiledTrace c : s.validOn) {
            if (!c.busy) {
                return c;
            }
        }
        return null;
    }

    /** Brings the newest device contents of {@code s} to its host array. */
    static void fetch(Storage s) {
        LOCK.lock();
        try {
            awaitWritten(s, null);
            if (s.hostValid) {
                return;
            }
            transfers[0]++;
            transfers[1] += s.numel;
            if (s.persistent) {
                if (s.validOn.isEmpty()) {
                    throw new IllegalStateException("persistent storage " + s.id + " has no valid copy");
                }
                // a plan another thread is running is not read from: wait until one of the copies is idle
                CompiledTrace ct;
                while ((ct = idleCopy(s)) == null) {
                    WRITTEN.awaitUninterruptibly();
                    if (s.hostValid) {
                        return;
                    }
                    if (s.validOn.isEmpty()) {
                        throw new IllegalStateException("persistent storage " + s.id + " has no valid copy");
                    }
                }
                ct.last.transferToHost(s.array);
                s.hostValid = true;
                return;
            }
            CompiledTrace ct = s.boundTrace;
            if (ct == null || s.stale || ct.bound[indexOf(ct, s)] != s || s.boundExec != ct.executions) {
                throw new IllegalStateException("tensor data was reused by a later execution of the same trace; "
                        + "copy what you need (toFloatArray) before the next step");
            }
            ct.last.transferToHost(s.array);
            s.hostValid = true;
        } finally {
            LOCK.unlock();
        }
    }

    /**
     * The intermediate slot of {@code ct} that holds {@code s} from the plan's previous execution
     * (its buffer still has the value on the device), or -1.
     */
    static int previousOutput(CompiledTrace ct, Storage s) {
        if (s.boundTrace != ct || s.stale || s.boundExec != ct.executions) {
            return -1;
        }
        for (int j = 0; j < ct.bound.length; j++) {
            if (ct.bound[j] == s && ct.kinds[j] == CompiledTrace.INTERMEDIATE && (ct.recyclable == null || !ct.recyclable[j])) {
                return j;
            }
        }
        return -1;
    }

    static int indexOf(CompiledTrace ct, Storage s) {
        for (int i = 0; i < ct.bound.length; i++) {
            if (ct.bound[i] == s) {
                return i;
            }
        }
        return 0;
    }

    /** Inputs from this size are copied into a plan's buffer by several threads. */
    static final long PARALLEL_COPY_BYTES = JitnnConfig.current().get(JitnnConfig.PARALLEL_COPY_BYTES);
    static final int COPY_THREADS = Math.min(8, Runtime.getRuntime().availableProcessors());

    /**
     * Copies an input into a plan's buffer. One thread copied ResNet-50's [128, 3, 224, 224] FP32
     * image (77 MB) in 6.3 ms, 8% of the call; large inputs are split across the common pool.
     */
    static void stage(MemorySegment from, MemorySegment to, long bytes) {
        if (bytes < PARALLEL_COPY_BYTES || COPY_THREADS < 2) {
            MemorySegment.copy(from, 0, to, 0, bytes);
            return;
        }
        long chunk = ((bytes + COPY_THREADS - 1) / COPY_THREADS + 63) & ~63L;
        java.util.stream.IntStream.range(0, COPY_THREADS).parallel().forEach(c -> {
            long start = c * chunk;
            long end = Math.min(bytes, start + chunk);
            if (start < end) {
                MemorySegment.copy(from, start, to, start, end - start);
            }
        });
    }

    static void copy(TornadoNativeArray from, TornadoNativeArray to) {
        if (from == to) {
            return;
        }
        MemorySegment.copy(from.getSegment(), 0, to.getSegment(), 0, Math.min(from.getNumBytesOfSegment(), to.getNumBytesOfSegment()));
    }
}
