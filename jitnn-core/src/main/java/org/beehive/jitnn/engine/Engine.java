package org.beehive.jitnn.engine;

import java.util.List;
import java.util.function.Supplier;

import org.beehive.jitnn.Tensor;

/**
 * Where operators run.
 *
 * <ul>
 * <li>{@link Backend#TORNADO}: operators are recorded lazily; a sync point compiles everything
 * recorded since the last one into one TornadoVM task graph, caches the compiled plan by the
 * trace's structure, and replays it when the same structure comes back (a training step).</li>
 * <li>{@link Backend#REFERENCE}: operators run eagerly on the host with the Java reference
 * implementations. Tests use it as the oracle; it needs no TornadoVM runtime.</li>
 * </ul>
 *
 * <p>The default is TORNADO when a TornadoVM runtime is on the class path, else REFERENCE; the
 * {@code jitnn.backend} system property ({@code tornado} or {@code reference}) overrides it.
 */
public final class Engine {

    public enum Backend {
        TORNADO,
        REFERENCE
    }

    private static final ThreadLocal<Backend> BACKEND = ThreadLocal.withInitial(Engine::defaultBackend);

    private Engine() {
    }

    private static Backend defaultBackend() {
        String p = System.getProperty("jitnn.backend");
        if (p != null) {
            return Backend.valueOf(p.trim().toUpperCase());
        }
        try {
            Class.forName("uk.ac.manchester.tornado.runtime.TornadoCoreRuntime", false, Engine.class.getClassLoader());
            return Backend.TORNADO;
        } catch (ClassNotFoundException e) {
            return Backend.REFERENCE;
        }
    }

    public static Backend backend() {
        return BACKEND.get();
    }

    /** True when this thread's plans run through TornadoVM on a CUDA device. */
    /**
     * Whether a kernel needing {@code bytes} of local memory per work-group fits the device plans run
     * on (Metal allows 32 KB, CUDA 48 KB); true when the device does not report a size.
     */
    public static boolean localMemoryFits(long bytes) {
        long available = Devices.localMemory();
        return available <= 0 || bytes <= available;
    }

    public static boolean onCuda() {
        return backend() == Backend.TORNADO && Devices.backend() == uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.CUDA;
    }

    /** Switches this thread's backend, first running anything recorded on the old one. */
    public static void setBackend(Backend backend) {
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
     * through the host, as {@link org.beehive.jitnn.dist.HostCollective} does).
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

    /** Records a node: executes it now on REFERENCE, defers it on TORNADO. */
    public static void record(Node node) {
        if (backend() == Backend.REFERENCE) {
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
        List<Node> nodes = Trace.drain();
        if (!nodes.isEmpty()) {
            TornadoExecutor.run(nodes, wanted);
        }
        for (Tensor t : wanted) {
            ensureHost(t.storage());
        }
    }

    /**
     * Runs {@code body} and returns the operators it recorded on this thread without running them
     * (anything recorded before is run first). For tools that inspect or replay a model's
     * operators, such as the kernel benchmark; the returned operators' outputs are never computed.
     */
    public static List<Node> capture(Runnable body) {
        sync();
        body.run();
        return Trace.drain();
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

    /** Plans compiled, replayed and reused for released storages on this JVM so far: {@code [compiles, replays, reuses]}. */
    public static long[] planStats() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.stats() : new long[3];
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
     * Cumulative nanoseconds spent in jitNN's executor outside TornadoVM ({@code [0]}, tracing
     * excluded) and inside {@code TornadoExecutionPlan.execute()} ({@code [1]}).
     */
    /**
     * Makes room for at least {@code plans} more compiled plans in the plan cache (per concurrency
     * slot), beyond {@code jitnn.plan.cache}: for callers that cycle through many shapes, as shape
     * buckets do, so their plans are not evicted and compiled again.
     */
    public static void reservePlans(int plans) {
        TornadoExecutor.RESERVED.addAndGet(Math.max(0, plans));
    }

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
     * Host-device traffic of this JVM's plans so far: [fetches to the host, elements fetched,
     * re-uploads of persistent storages written by another plan, elements re-uploaded, input
     * copies, elements copied].
     */
    /**
     * Totals over every plan execution so far: {@code [executions, kernel launches, library calls,
     * bytes]}, the bytes being what the operators read and write at least (each input and output
     * tensor once: a lower bound on device-memory traffic that fusion lowers). Differences between
     * two calls give a step's counts.
     */
    public static long[] trafficStats() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.trafficStats() : new long[4];
    }

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
