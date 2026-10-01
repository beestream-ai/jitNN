package org.beehive.jitnn.engine;

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

import org.beehive.jitnn.Tensor;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
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
 * compiling one.
 */
final class TornadoExecutor {

    private static final int CACHE_SIZE = Integer.getInteger("jitnn.plan.cache", 16);
    private static final boolean VERBOSE = Boolean.getBoolean("jitnn.verbose");
    private static final boolean PROFILE = Boolean.getBoolean("jitnn.profile");
    /** On CUDA, capture each plan as a CUDA graph and replay it (default on). */
    private static final boolean CUDA_GRAPH = Boolean.parseBoolean(System.getProperty("jitnn.cudaGraph", "true"));
    private static long lastKernelNanos;
    private static long hostNanos;
    private static long executeNanos;
    private static long lastTotalNanos;
    private static final AtomicInteger GRAPHS = new AtomicInteger();
    private static volatile boolean loaded;

    private static final Map<String, CompiledTrace> CACHE = new LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, CompiledTrace> eldest) {
            if (size() > CACHE_SIZE) {
                close(eldest.getValue());
                return true;
            }
            return false;
        }
    };

    private static long compiles;
    private static long replays;

    private TornadoExecutor() {
    }

    static boolean isLoaded() {
        return loaded;
    }

    static synchronized void run(List<Node> nodes, Tensor[] wanted) {
        long start = System.nanoTime();
        long before = executeNanos;
        loaded = true;
        // Slot table over storages, in order of first appearance.
        IdentityHashMap<Storage, Integer> slotOf = new IdentityHashMap<>();
        List<Storage> slots = new ArrayList<>();
        Set<Storage> produced = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Node n : nodes) {
            for (Tensor t : n.inputs) {
                slotOf.computeIfAbsent(t.storage(), s -> add(slots, s));
            }
            for (Tensor t : n.outputs) {
                slotOf.computeIfAbsent(t.storage(), s -> add(slots, s));
                produced.add(t.storage());
            }
        }
        int[] kinds = new int[slots.size()];
        boolean[] written = new boolean[slots.size()];
        StringBuilder key = new StringBuilder(256).append(Devices.key()).append('|');
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (s.persistent) {
                kinds[i] = CompiledTrace.PERSISTENT;
                written[i] = produced.contains(s);
                key.append('P').append(s.id).append(written[i] ? 'w' : 'r');
            } else if (produced.contains(s)) {
                kinds[i] = CompiledTrace.INTERMEDIATE;
                key.append('T');
            } else {
                kinds[i] = CompiledTrace.INPUT;
                key.append('I');
            }
            key.append(s.dtype.ordinal()).append(':').append(s.numel).append(',');
        }
        for (Node n : nodes) {
            key.append('|').append(n.op.name()).append(Arrays.toString(n.iattr)).append(Arrays.toString(n.fattr));
            for (Tensor t : n.inputs) {
                key.append('<').append(slotOf.get(t.storage())).append(t.shape());
            }
            for (Tensor t : n.outputs) {
                key.append('>').append(slotOf.get(t.storage())).append(t.shape());
            }
        }
        String k = key.toString();
        CompiledTrace ct = CACHE.get(k);
        if (ct == null) {
            ct = compile(k, nodes, slots, slotOf, kinds, written);
            CACHE.put(k, ct);
            compiles++;
        } else {
            replays++;
        }
        execute(ct, slots);
        for (Node n : nodes) {
            n.executed = true;
        }
        for (Tensor t : wanted) {
            Storage s = t.storage();
            if (!s.hostValid && slotOf.containsKey(s)) {
                fetch(s);
            }
        }
        hostNanos += System.nanoTime() - start - (executeNanos - before);
    }

    private static int add(List<Storage> slots, Storage s) {
        slots.add(s);
        return slots.size() - 1;
    }

    private static CompiledTrace compile(String key, List<Node> nodes, List<Storage> slots, IdentityHashMap<Storage, Integer> slotOf, int[] kinds, boolean[] written) {
        long t0 = System.nanoTime();
        String name = "jitnn" + GRAPHS.incrementAndGet();
        Object[] arrays = new Object[slots.size()];
        Storage[] bound = new Storage[slots.size()];
        IdentityHashMap<Storage, Object> byStorage = new IdentityHashMap<>();
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            arrays[i] = switch (kinds[i]) {
                case CompiledTrace.PERSISTENT -> s.allocate();
                default -> Storage.newArray(s.dtype, s.numel);
            };
            if (kinds[i] == CompiledTrace.PERSISTENT) {
                bound[i] = s;
            }
            byStorage.put(s, arrays[i]);
        }
        Emitter e = new Emitter(name, byStorage);
        for (Node n : nodes) {
            n.op.emit(e, n);
        }

        Devices.pin(name);
        TaskGraph graph = new TaskGraph(name);
        List<Object> inputs = new ArrayList<>();
        List<Object> once = new ArrayList<>(e.constants);
        List<Object> keep = new ArrayList<>();
        for (int i = 0; i < slots.size(); i++) {
            switch (kinds[i]) {
                case CompiledTrace.INPUT -> inputs.add(arrays[i]);
                case CompiledTrace.PERSISTENT -> {
                    once.add(arrays[i]);
                    keep.add(arrays[i]);
                }
                default -> keep.add(arrays[i]);
            }
        }
        if (!inputs.isEmpty()) {
            graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs.toArray());
        }
        if (!once.isEmpty()) {
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, once.toArray());
        }
        for (Emitter.Spec spec : e.tasks) {
            spec.adder().add(graph, spec.id());
        }
        if (!keep.isEmpty()) {
            graph.persistOnDevice(keep.toArray());
        }
        GridScheduler scheduler = new GridScheduler();
        e.registerGrids(scheduler);

        CompiledTrace ct = new CompiledTrace(key, name, kinds, arrays, written, bound, e.tasks.size());
        TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot());
        plan.withGridScheduler(scheduler);
        if (PROFILE) {
            plan.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.SILENT);
        }
        Devices.apply(plan);
        if (CUDA_GRAPH && Devices.backend() == uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.CUDA) {
            plan.withCUDAGraph();
        }
        ct.plan = plan;
        ct.compileNanos = System.nanoTime() - t0;
        if (VERBOSE) {
            System.err.printf("[jitnn] compiled %s on %s: %d nodes, %d tasks, %d slots, %.1f ms%n", name, plan.getDevice(0).getTornadoVMBackend() + " " + plan.getDevice(0).getDeviceName(),
                    nodes.size(), e.tasks.size(), slots.size(), ct.compileNanos / 1e6);
        }
        return ct;
    }

    private static void execute(CompiledTrace ct, List<Storage> slots) {
        // Inputs first: an input may be an intermediate of this plan's previous execution, which
        // must be fetched before its slot is rebound below.
        for (int i = 0; i < slots.size(); i++) {
            if (ct.kinds[i] == CompiledTrace.INPUT) {
                Storage s = slots.get(i);
                Engine.ensureHost(s);
                copy((TornadoNativeArray) s.array, (TornadoNativeArray) ct.arrays[i]);
            }
        }
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (ct.kinds[i] == CompiledTrace.PERSISTENT && !s.validOn.contains(ct)) {
                if (!s.hostValid) {
                    fetch(s);
                }
                if (ct.executions > 0) {
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
                if (previous.hostValid) {
                    // Keep the host copy the user may still read: detach it from the plan's array.
                    TornadoNativeArray own = Storage.newArray(previous.dtype, previous.numel);
                    copy((TornadoNativeArray) previous.array, own);
                    previous.array = own;
                } else {
                    previous.stale = true;
                }
                previous.boundTrace = null;
            }
            s.array = (TornadoNativeArray) ct.arrays[i];
            ct.bound[i] = s;
        }
        long t0 = System.nanoTime();
        ct.last = ct.plan.execute();
        executeNanos += System.nanoTime() - t0;
        ct.executions++;
        if (PROFILE) {
            lastKernelNanos = ct.last.getProfilerResult().getDeviceKernelTime();
            lastTotalNanos = ct.last.getProfilerResult().getTotalTime();
        }
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (ct.kinds[i] == CompiledTrace.PERSISTENT) {
                if (ct.written[i]) {
                    s.validOn.clear();
                    s.hostValid = false;
                }
                s.validOn.add(ct);
            } else if (ct.kinds[i] == CompiledTrace.INTERMEDIATE) {
                s.boundTrace = ct;
                s.boundExec = ct.executions;
                s.hostValid = false;
                s.stale = false;
            }
        }
    }

    /** Brings the newest device contents of {@code s} to its host array. */
    static synchronized void fetch(Storage s) {
        if (s.hostValid) {
            return;
        }
        if (s.persistent) {
            if (s.validOn.isEmpty()) {
                throw new IllegalStateException("persistent storage " + s.id + " has no valid copy");
            }
            CompiledTrace ct = s.validOn.iterator().next();
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
    }

    private static int indexOf(CompiledTrace ct, Storage s) {
        for (int i = 0; i < ct.bound.length; i++) {
            if (ct.bound[i] == s) {
                return i;
            }
        }
        return 0;
    }

    static void copy(TornadoNativeArray from, TornadoNativeArray to) {
        if (from == to) {
            return;
        }
        MemorySegment.copy(from.getSegment(), 0, to.getSegment(), 0, Math.min(from.getNumBytesOfSegment(), to.getNumBytesOfSegment()));
    }

    private static void close(CompiledTrace ct) {
        for (int i = 0; i < ct.kinds.length; i++) {
            Storage s = ct.bound[i];
            if (s == null) {
                continue;
            }
            if (ct.kinds[i] == CompiledTrace.PERSISTENT) {
                if (s.validOn.contains(ct) && !s.hostValid && s.validOn.size() == 1) {
                    ct.last.transferToHost(s.array);
                    s.hostValid = true;
                }
                s.validOn.remove(ct);
            } else if (s.boundTrace == ct && !s.hostValid) {
                s.stale = true;
            }
        }
        try {
            ct.plan.close();
        } catch (TornadoExecutionPlanException e) {
            throw new IllegalStateException(e);
        }
    }

    static synchronized void closeAll() {
        for (CompiledTrace ct : CACHE.values()) {
            close(ct);
        }
        CACHE.clear();
    }

    /** With -Djitnn.profile=true: [device kernel ns, TornadoVM total ns] of the last execution. */
    static synchronized long[] lastProfile() {
        return new long[] { lastKernelNanos, lastTotalNanos };
    }

    /** Cumulative [jitNN host ns outside plan.execute, ns inside plan.execute] of run(). */
    static synchronized long[] timeSplit() {
        return new long[] { hostNanos, executeNanos };
    }

    /** Compile and replay counts, for tests and benchmarks. */
    static synchronized long[] stats() {
        return new long[] { compiles, replays };
    }
}
