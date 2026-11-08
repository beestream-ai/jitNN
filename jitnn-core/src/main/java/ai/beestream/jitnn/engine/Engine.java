package ai.beestream.jitnn.engine;

import java.util.List;
import java.util.function.Supplier;

import ai.beestream.jitnn.Tensor;

/**
 * Where operators run.
 *
 * <ul>
 * <li>{@link Backend#TORNADO}: operators are recorded lazily; a sync point compiles everything
 * recorded since the last one into one TornadoVM task graph, caches the compiled plan by the
 * trace's structure, and replays it when the same structure comes back (a training step).</li>
 * <li>{@link Backend#CPU}: operators are recorded lazily as on TORNADO; a sync point rewrites the
 * trace with the same fusion pass and runs it on the host, each operator with its multithreaded,
 * vectorised CPU kernel (Java Vector API) or, when it has none, its reference implementation. It
 * needs no TornadoVM runtime, but the JVM needs {@code --add-modules jdk.incubator.vector}.</li>
 * <li>{@link Backend#REFERENCE}: operators run eagerly on the host with the Java reference
 * implementations. Tests use it as the oracle; it needs no TornadoVM runtime.</li>
 * </ul>
 *
 * <p>The default is TORNADO when a TornadoVM runtime is on the class path, else CPU when the JVM has
 * the Vector API module, else REFERENCE; the {@code jitnn.backend} system property ({@code tornado},
 * {@code cpu} or {@code reference}) overrides it.
 */
public final class Engine {

    /** Where operators run: see the class description. */
    public enum Backend {
        /** Lazily, through compiled TornadoVM plans on the selected device. */
        TORNADO,
        CPU,
        /** Eagerly, on the host, with the Java reference implementations. */
        REFERENCE
    }

    private static final ThreadLocal<Backend> BACKEND = ThreadLocal.withInitial(Engine::defaultBackend);

    private Engine() {
    }

    private static Backend defaultBackend() {
        String p = System.getProperty("jitnn.backend");
        if (p != null) {
            return checked(Backend.valueOf(p.trim().toUpperCase(java.util.Locale.ROOT)));
        }
        try {
            Class.forName("uk.ac.manchester.tornado.runtime.TornadoCoreRuntime", false, Engine.class.getClassLoader());
            return Backend.TORNADO;
        } catch (ClassNotFoundException e) {
            return VECTOR_API ? Backend.CPU : Backend.REFERENCE;
        }
    }

    /** True when the JVM resolved the Vector API module the CPU backend's kernels use. */
    private static final boolean VECTOR_API = ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();

    private static Backend checked(Backend b) {
        if (b == Backend.CPU && !VECTOR_API) {
            throw new IllegalStateException("the CPU backend needs the Vector API: run the JVM with --add-modules jdk.incubator.vector");
        }
        return b;
    }

    /** This thread's backend. */
    public static Backend backend() {
        return BACKEND.get();
    }

    /**
     * Whether a kernel needing {@code bytes} of local memory per work-group fits the device plans run
     * on (Metal allows 32 KB, CUDA 48 KB); true when the device does not report a size.
     */
    public static boolean localMemoryFits(long bytes) {
        long available = Devices.localMemory();
        return available <= 0 || bytes <= available;
    }

    /** True when this thread's plans run through TornadoVM on a CUDA device. */
    public static boolean onCuda() {
        return backend() == Backend.TORNADO && Devices.backend() == uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.CUDA;
    }

    /** Switches this thread's backend, first running anything recorded on the old one. */
    public static void setBackend(Backend backend) {
        checked(backend);
        sync();
        BACKEND.set(backend);
    }

    /** Runs {@code body} on {@code backend} and restores the previous backend. */
    public static <T> T with(Backend backend, Supplier<T> body) {
        Backend previous = backend();
        setBackend(backend);
        try {
            T result = body.get();
            sync();
            return result;
        } finally {
            BACKEND.set(previous);
        }
    }

    /**
     * Runs {@code body} with this thread's plans on device {@code spec} ("driver:device", as
     * {@code -Djitnn.device}), then restores the previous device. Tensors belong to the device
     * they were computed on: do not share them between threads on different devices (move data
     * through the host, as {@link ai.beestream.jitnn.dist.HostCollective} does).
     */
    public static <T> T onDevice(String spec, Supplier<T> body) {
        String previous = Devices.current();
        Devices.set(spec);
        try {
            T result = body.get();
            sync();
            return result;
        } finally {
            Devices.set(previous);
        }
    }

    /** Set while {@link #capture} records: operators are kept, never run, on every backend. */
    private static final ThreadLocal<Boolean> CAPTURING = ThreadLocal.withInitial(() -> Boolean.FALSE);

    /** Records a node: executes it now on REFERENCE, defers it to the next sync on TORNADO and CPU. */
    public static void record(Node node) {
        if (backend() == Backend.REFERENCE && !CAPTURING.get()) {
            ReferenceExecutor.run(node);
        } else {
            Trace.record(node);
        }
    }

    /**
     * Computes {@code t} on the host when it is the output of the last recorded operator and that
     * operator draws uniform random values from a seed ({@code Tensor.uniform(long, ...)}): the
     * operator leaves the trace, so no device plan is compiled for it. Returns true when it did.
     * Initial weights use this: the host's uniform stream equals the device's bit for bit (normal
     * draws are left to the device, whose transcendentals are FP32).
     */
    public static boolean generateOnHost(Tensor t) {
        java.util.List<Node> pending = Trace.pending();
        if (pending.isEmpty()) {
            return false;
        }
        Node last = pending.get(pending.size() - 1);
        if (last.inputs.length != 0 || last.outputs.length != 1 || last.outputs[0].storage() != t.storage() || !last.op.name().equals("philox")
                || last.iattr[3] != 0 || !Trace.removeLast(last)) {
            return false;
        }
        ReferenceExecutor.run(last);
        return true;
    }

    /** Runs everything recorded on this thread, fetching {@code wanted} to the host. */
    public static void sync(Tensor... wanted) {
        if (CAPTURING.get()) {
            throw new IllegalStateException("the code being captured reads a tensor, which would run the operators being captured");
        }
        List<Node> nodes = Trace.drain();
        if (!nodes.isEmpty()) {
            if (backend() == Backend.CPU) {
                CpuExecutor.run(nodes, wanted);
            } else {
                TornadoExecutor.run(nodes, wanted);
            }
        }
        for (Tensor t : wanted) {
            ensureHost(t.storage());
        }
    }

    /**
     * Runs {@code body} and returns the operators it recorded on this thread without running them
     * (anything recorded before is run first), on either backend. For tools that inspect, replay
     * or export a model's operators, such as the kernel benchmark and the ONNX export; the returned
     * operators' outputs are never computed, and reading a tensor inside {@code body} is an error.
     */
    public static List<Node> capture(Runnable body) {
        sync();
        CAPTURING.set(Boolean.TRUE);
        try {
            body.run();
            return Trace.drain();
        } finally {
            CAPTURING.set(Boolean.FALSE);
            Trace.drain();
        }
    }

    /**
     * The CPU backend's counters so far, per operator name: {@code [calls, nanoseconds, calls that
     * ran the reference implementation, calls that ran on the host's OpenCL device]}; the times only
     * with {@code -Djitnn.cpu.stats=true}, which also prints the table at exit.
     */
    public static java.util.Map<String, long[]> cpuStats() {
        return CpuExecutor.stats();
    }

    /**
     * The OpenCL device on the host CPU ({@code <driver>:<device>}) where the CPU backend runs the
     * operators it has no kernel for, or null when there is none (no TornadoVM runtime, no OpenCL CPU
     * driver, {@code -Djitnn.cpu.opencl=off}); those operators then run their reference.
     */
    public static String cpuOpenClDevice() {
        return Devices.hostOpenCl();
    }

    /**
     * The CPU backend's operators by time since the last call (as printed at exit with
     * {@code jitnn.cpu.stats}), then starts counting again; empty when nothing ran.
     */
    public static String cpuReport() {
        return CpuExecutor.drainReport();
    }

    /** Number of operators recorded and not yet run on this thread. */
    public static int pending() {
        return Trace.size();
    }

    static void ensureHost(Storage s) {
        if (s.dropped) {
            throw new IllegalStateException("tensor data is gone: the tensor was released (Tensor.release or close)");
        }
        if (s.stale) {
            throw new IllegalStateException("tensor data is gone: its buffer was reused, by a later execution of the same trace or by a later "
                    + "tensor of the same trace once nothing needed it; pass the tensor to the sync that runs it (Engine.sync(t), t.item()) or read it before "
                    + "the next step");
        }
        if (s.hostValid) {
            return;
        }
        if (Trace.size() > 0) {
            sync();
            if (s.hostValid) {
                return;
            }
        }
        TornadoExecutor.fetch(s);
    }

    /**
     * Plans compiled, replayed and reused for released storages on this JVM so far, and the replays
     * that skipped the rewrite (a trace remembered, {@code jitnn.plan.memo}):
     * {@code [compiles, replays, reuses, remembered]}.
     */
    public static long[] planStats() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.stats() : new long[4];
    }

    /**
     * Starts recording a timeline of plan executions for {@link #stopTrace}. Needs
     * {@code -Djitnn.profile=true} (TornadoVM's profiler gives each kernel's device time) and, on
     * CUDA, {@code -Djitnn.cudaGraph=false} (a CUDA graph reports one time for all its kernels).
     */
    public static void startTrace() {
        if (!Boolean.getBoolean("jitnn.profile")) {
            throw new IllegalStateException("a timeline needs -Djitnn.profile=true (and -Djitnn.cudaGraph=false on CUDA)");
        }
        TornadoExecutor.traceStart();
        tracing = true;
    }

    private static volatile boolean tracing;

    private static volatile boolean deterministic = initialDeterministic();

    private static boolean initialDeterministic() {
        boolean on = Boolean.getBoolean("jitnn.deterministic");
        if (on) {
            System.setProperty("tornado.cudnn.deterministic", "true");
        }
        return on;
    }

    /**
     * Run-to-run reproducible results ({@code -Djitnn.deterministic=true}), as PyTorch's
     * {@code use_deterministic_algorithms}: jitNN's own kernels are deterministic already (no
     * atomics; split reductions sum in a fixed order); this also stops autotuning from measuring
     * (choices follow the rules or the saved decisions, never the timings of this run) and asks
     * cuDNN for deterministic algorithms only. Same device, same library versions, same inputs:
     * the same bits.
     */
    public static void setDeterministic(boolean on) {
        deterministic = on;
        System.setProperty("tornado.cudnn.deterministic", Boolean.toString(on));
    }

    /** Whether deterministic kernels were asked for ({@link #setDeterministic}). */
    public static boolean deterministic() {
        return deterministic;
    }

    /**
     * Stops recording and writes the timeline as Chrome trace events (JSON), for Perfetto
     * (ui.perfetto.dev) or chrome://tracing. Thread "host": every plan execution, from the call to
     * its end (the first one compiles); thread "device": every kernel's device time, in launch order,
     * laid back to back from the start of its execution (TornadoVM reports durations only); thread
     * "ranges": the spans opened with {@link #traceRange}.
     */
    public static void stopTrace(java.nio.file.Path file) throws java.io.IOException {
        tracing = false;
        List<String> events = new java.util.ArrayList<>(TornadoExecutor.traceStop());
        events.add(0, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"ranges\"}}");
        events.add(1, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":1,\"args\":{\"name\":\"host: plan executions\"}}");
        events.add(2, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":2,\"args\":{\"name\":\"device: kernels\"}}");
        events.add(3, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{\"name\":\"jitNN\"}}");
        java.nio.file.Files.writeString(file, "{\"traceEvents\":[\n" + String.join(",\n", events) + "\n],\"displayTimeUnit\":\"ms\"}\n");
    }

    /**
     * A named span on the timeline (thread "ranges"), for example one training step, while
     * recording; nothing otherwise. Close it (try-with-resources) where the span ends.
     */
    public static AutoCloseable traceRange(String name) {
        if (!tracing) {
            return () -> {
            };
        }
        long start = System.nanoTime();
        return () -> TornadoExecutor.traceEvent(String.format(java.util.Locale.ROOT, "{\"name\":\"%s\",\"cat\":\"range\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":0}",
                name.replace("\"", "'"), (start - TornadoExecutor.traceOrigin()) / 1e3, (System.nanoTime() - start) / 1e3));
    }

    /**
     * Device memory for intermediates (activations, gradients and other temporaries, after buffer
     * reuse) of the last plan that ran, in bytes; 0 on the reference backend.
     */
    public static long lastIntermediateBytes() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.lastIntermediateBytes() : 0;
    }

    /**
     * With {@code -Djitnn.profile=true}: {@code [device kernel ns, TornadoVM total ns]} of the
     * last plan execution, from TornadoVM's profiler. Zeros otherwise.
     */
    public static long[] lastProfile() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.lastProfile() : new long[2];
    }

    /**
     * With -Djitnn.profile=true: device time per operator task ("op#k", k-th task of the operator)
     * as [launches, nanoseconds], accumulated since the last call (which resets it).
     */
    public static java.util.Map<String, long[]> kernelBreakdown() {
        return TornadoExecutor.kernelBreakdown();
    }

    /**
     * Makes room for at least {@code plans} more compiled plans in the plan cache (per concurrency
     * slot), beyond {@code jitnn.plan.cache}: for callers that cycle through many shapes, as shape
     * buckets do, so their plans are not evicted and compiled again.
     */
    public static void reservePlans(int plans) {
        TornadoExecutor.RESERVED.addAndGet(Math.max(0, plans));
    }

    /**
     * Cumulative nanoseconds spent in jitNN's executor outside TornadoVM ({@code [0]}, tracing
     * excluded) and inside {@code TornadoExecutionPlan.execute()} ({@code [1]}).
     */
    public static long[] timeSplit() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.timeSplit() : new long[2];
    }

    private static volatile java.io.PrintStream kernelSink = initialKernelSink();

    private static java.io.PrintStream initialKernelSink() {
        String p = System.getProperty("jitnn.printKernels", "").trim();
        if (p.isEmpty() || p.equals("false")) {
            return null;
        }
        if (p.equals("true")) {
            return System.out;
        }
        try {
            return new java.io.PrintStream(new java.io.FileOutputStream(p, true), true);
        } catch (java.io.FileNotFoundException e) {
            throw new IllegalArgumentException("jitnn.printKernels: cannot write " + p, e);
        }
    }

    /**
     * Prints every plan compiled from now on to {@code out} (null stops): the plan's operators with
     * their shapes and task counts, then the CUDA, OpenCL or Metal source TornadoVM generates for
     * each kernel, on the plan's first run. Plans already compiled are not printed again; call
     * {@link #reset()} first to recompile them. {@code -Djitnn.printKernels=true} (standard output)
     * or {@code =<file>} turns it on at start-up.
     *
     * <p>While a plan is compiled, standard output is redirected to collect TornadoVM's dump;
     * other threads' output in that window goes into the dump.
     */
    public static void printKernels(java.io.PrintStream out) {
        kernelSink = out;
    }

    static java.io.PrintStream kernelSink() {
        return kernelSink;
    }

    /**
     * Totals over every plan execution so far: {@code [executions, kernel launches, library calls,
     * bytes]}, the bytes being what the operators read and write at least (each input and output
     * tensor once: a lower bound on device-memory traffic that fusion lowers). Differences between
     * two calls give a step's counts.
     */
    public static long[] trafficStats() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.trafficStats() : new long[4];
    }

    /**
     * Host-device traffic of this JVM's plans so far: [fetches to the host, elements fetched,
     * re-uploads of persistent storages written by another plan, elements re-uploaded, input
     * copies, elements copied].
     */
    public static long[] transferStats() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.transferStats() : new long[6];
    }

    /** Closes every cached compiled plan, first saving persistent storages to the host. */
    public static void reset() {
        sync();
        if (TornadoExecutor.isLoaded()) {
            TornadoExecutor.closeAll();
        }
    }
}
