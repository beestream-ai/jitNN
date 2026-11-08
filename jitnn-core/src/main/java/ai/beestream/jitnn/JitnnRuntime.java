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
package ai.beestream.jitnn;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Supplier;

import ai.beestream.jitnn.engine.Autotune;
import ai.beestream.jitnn.engine.Engine;

/**
 * jitNN's runtime: where operators run, sync points, determinism, profiling and statistics.
 *
 * <p>Operators on {@link Backend#TORNADO} and {@link Backend#CPU} are recorded lazily and run at a
 * sync point: reading a tensor ({@link Tensor#item()}, {@link Tensor#toFloatArray()}) or
 * {@link #sync}. Compiled plans are cached by the trace's structure, so a training step or an
 * inference call with the same shapes replays its plan. Options are set with {@link JitnnConfig}.
 *
 * <p>The backend and the device are per thread; everything else is per JVM.
 */
public final class JitnnRuntime {

    private JitnnRuntime() {
    }

    /** This thread's backend. */
    public static Backend backend() {
        return Engine.backend();
    }

    /** Switches this thread's backend, first running anything recorded on the old one. */
    public static void setBackend(Backend backend) {
        Engine.setBackend(backend);
    }

    /** Runs {@code body} on {@code backend}, syncs, and restores the previous backend. */
    public static <T> T with(Backend backend, Supplier<T> body) {
        return Engine.with(backend, body);
    }

    /**
     * Runs {@code body} with this thread's plans on device {@code spec} ({@code "driver:device"},
     * as {@code jitnn.device}), syncs, then restores the previous device. Tensors belong to the
     * device they were computed on: do not share them between threads on different devices (move
     * data through the host, as {@link ai.beestream.jitnn.dist.HostCollective} does).
     */
    public static <T> T onDevice(String spec, Supplier<T> body) {
        return Engine.onDevice(spec, body);
    }

    /** True when this thread's plans run through TornadoVM on a CUDA device. */
    public static boolean onCuda() {
        return Engine.onCuda();
    }

    /** Runs everything recorded on this thread, fetching {@code wanted} to the host. */
    public static void sync(Tensor... wanted) {
        Engine.sync(wanted);
    }

    /** Number of operators recorded and not yet run on this thread. */
    public static int pending() {
        return Engine.pending();
    }

    /**
     * Closes every cached compiled plan, first saving persistent storages (weights) to the host;
     * the next run of each trace compiles again.
     */
    public static void reset() {
        Engine.reset();
    }

    /**
     * Makes room for at least {@code plans} more compiled plans in the plan cache (per concurrency
     * slot), beyond {@code jitnn.plan.cache}: for callers that cycle through many shapes, as shape
     * buckets do, so their plans are not evicted and compiled again.
     */
    public static void reservePlans(int plans) {
        Engine.reservePlans(plans);
    }

    /**
     * Run-to-run reproducible results, as PyTorch's {@code use_deterministic_algorithms}: jitNN's
     * own kernels are deterministic already (no atomics; split reductions sum in a fixed order);
     * this also stops autotuning from measuring and asks cuDNN for deterministic algorithms only.
     * Same device, same library versions, same inputs: the same bits.
     */
    public static void setDeterministic(boolean on) {
        Engine.setDeterministic(on);
    }

    /** Whether deterministic results were asked for ({@link #setDeterministic}, {@code jitnn.deterministic}). */
    public static boolean deterministic() {
        return Engine.deterministic();
    }

    /** Sets run-time autotuning: {@code "on"} measures new shapes, {@code "use"} only applies saved decisions, {@code "off"}. */
    public static void setAutotune(String mode) {
        Autotune.setMode(mode);
    }

    /** Sets where autotuning decisions are saved; decisions loaded so far are dropped. */
    public static void setAutotuneDirectory(Path dir) {
        Autotune.setDirectory(dir);
    }

    /**
     * Prints every plan compiled from now on to {@code out} (null stops): the plan's operators with
     * their shapes and task counts, then the CUDA, OpenCL or Metal source of each kernel. Plans
     * already compiled are not printed again; call {@link #reset()} first to recompile them.
     */
    public static void printKernels(PrintStream out) {
        Engine.printKernels(out);
    }

    /**
     * Plans compiled, replayed and reused for released storages on this JVM so far, and the replays
     * that skipped the rewrite: {@code [compiles, replays, reuses, remembered]}.
     */
    public static long[] planStats() {
        return Engine.planStats();
    }

    /**
     * Host-device traffic of this JVM's plans so far: {@code [fetches to the host, elements
     * fetched, re-uploads of persistent storages written by another plan, elements re-uploaded,
     * input copies, elements copied]}.
     */
    public static long[] transferStats() {
        return Engine.transferStats();
    }

    /**
     * Totals over every plan execution so far: {@code [executions, kernel launches, library calls,
     * bytes]}, the bytes being what the operators read and write at least (a lower bound on
     * device-memory traffic that fusion lowers). Differences between two calls give a step's counts.
     */
    public static long[] trafficStats() {
        return Engine.trafficStats();
    }

    /**
     * Cumulative nanoseconds spent in jitNN's executor outside TornadoVM ({@code [0]}) and inside
     * TornadoVM's plan execution ({@code [1]}).
     */
    public static long[] timeSplit() {
        return Engine.timeSplit();
    }

    /**
     * Device memory for intermediates (activations, gradients and other temporaries, after buffer
     * reuse) of the last plan that ran, in bytes; 0 on the reference backend.
     */
    public static long lastIntermediateBytes() {
        return Engine.lastIntermediateBytes();
    }

    /**
     * With {@code jitnn.profile=true}: {@code [device kernel ns, TornadoVM total ns]} of the last
     * plan execution, from TornadoVM's profiler. Zeros otherwise.
     */
    public static long[] lastProfile() {
        return Engine.lastProfile();
    }

    /**
     * With {@code jitnn.profile=true}: device time per operator task ({@code "op#k"}, the k-th task
     * of the operator) as {@code [launches, nanoseconds]}, accumulated since the last call, which
     * resets it.
     */
    public static Map<String, long[]> kernelBreakdown() {
        return Engine.kernelBreakdown();
    }

    /**
     * Starts recording a timeline of plan executions for {@link #stopTrace}. Needs
     * {@code jitnn.profile=true} and, on CUDA, {@code jitnn.cudaGraph=false}.
     */
    public static void startTrace() {
        Engine.startTrace();
    }

    /**
     * Stops recording and writes the timeline as Chrome trace events (JSON), for Perfetto or
     * chrome://tracing: plan executions, kernels, and the spans opened with {@link #traceRange}.
     */
    public static void stopTrace(Path file) throws IOException {
        Engine.stopTrace(file);
    }

    /**
     * A named span on the timeline (one training step, say) while recording; nothing otherwise.
     * Close it (try-with-resources) where the span ends.
     */
    public static AutoCloseable traceRange(String name) {
        return Engine.traceRange(name);
    }

    /**
     * The CPU backend's counters so far, per operator name: {@code [calls, nanoseconds, calls that
     * ran the reference implementation, calls that ran on the host's OpenCL device]}; the times only
     * with {@code jitnn.cpu.stats=true}.
     */
    public static Map<String, long[]> cpuStats() {
        return Engine.cpuStats();
    }

    /**
     * The CPU backend's operators by time since the last call (as printed at exit with
     * {@code jitnn.cpu.stats}), then starts counting again; empty when nothing ran.
     */
    public static String cpuReport() {
        return Engine.cpuReport();
    }

    /**
     * The OpenCL device on the host CPU ({@code <driver>:<device>}) where the CPU backend runs the
     * operators it has no kernel for, or null when there is none.
     */
    public static String cpuOpenClDevice() {
        return Engine.cpuOpenClDevice();
    }
}
