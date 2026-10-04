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

    /** Records a node: executes it now on REFERENCE, defers it on TORNADO. */
    public static void record(Node node) {
        if (backend() == Backend.REFERENCE) {
            ReferenceExecutor.run(node);
        } else {
            Trace.record(node);
        }
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

    /** Number of operators recorded and not yet run on this thread. */
    public static int pending() {
        return Trace.size();
    }

    static void ensureHost(Storage s) {
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

    /** Plans compiled and plans replayed on this JVM so far: {@code [compiles, replays]}. */
    public static long[] planStats() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.stats() : new long[2];
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
    public static long[] timeSplit() {
        return TornadoExecutor.isLoaded() ? TornadoExecutor.timeSplit() : new long[2];
    }

    /** Closes every cached compiled plan, first saving persistent storages to the host. */
    public static void reset() {
        sync();
        if (TornadoExecutor.isLoaded()) {
            TornadoExecutor.closeAll();
        }
    }
}
