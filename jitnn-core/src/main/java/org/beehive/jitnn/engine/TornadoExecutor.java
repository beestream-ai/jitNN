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
    /**
     * Tasks per TornadoVM task graph before the trace is split into a chain of graphs. A graph's
     * high-level bytecode buffer is 8 KiB and a task takes 15 + 5 bytes per argument (jitNN
     * kernels take up to 19), next to the transfer declarations: 40 tasks stays inside it.
     */
    private static final int MAX_TASKS = Integer.getInteger("jitnn.graph.maxTasks", 40);
    /** Reuse dead intermediates' buffers within a compiled trace (default on). */
    private static final boolean REUSE = Boolean.parseBoolean(System.getProperty("jitnn.reuseBuffers", "true"));
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
        // Buffer reuse (N4): an internal intermediate (a backward-pass temporary no user code can
        // hold) that a later node consumes and that was not requested at this sync hands its
        // buffer on once its last reader has run.
        int[] lastUse = new int[slots.size()];
        boolean[] consumed = new boolean[slots.size()];
        Arrays.fill(lastUse, -1);
        for (int ni = 0; ni < nodes.size(); ni++) {
            Node n = nodes.get(ni);
            for (Tensor t : n.inputs) {
                int si = slotOf.get(t.storage());
                lastUse[si] = ni;
                consumed[si] = true;
            }
            for (Tensor t : n.outputs) {
                int si = slotOf.get(t.storage());
                lastUse[si] = Math.max(lastUse[si], ni);
            }
        }
        boolean[] recyclable = new boolean[slots.size()];
        Set<Storage> wantedSet = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor t : wanted) {
            wantedSet.add(t.storage());
        }
        if (REUSE) {
            key.append("|R");
            for (int i = 0; i < slots.size(); i++) {
                recyclable[i] = kinds[i] == CompiledTrace.INTERMEDIATE && consumed[i] && slots.get(i).internal && !wantedSet.contains(slots.get(i));
                if (kinds[i] == CompiledTrace.INTERMEDIATE && !recyclable[i]) {
                    key.append(i).append(',');
                }
            }
        }
        String k = key.toString();
        CompiledTrace ct = CACHE.get(k);
        if (ct == null) {
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse);
            CACHE.put(k, ct);
            compiles++;
        } else {
            replays++;
        }
        try {
            execute(ct, slots);
        } catch (uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException oom) {
            // Device memory is shared by every cached plan. WORKAROUND(TornadoVM J17): a plan
            // whose execution failed is not safe to execute again, so it is dropped, the other
            // plans are closed (persistent storages saved to the host first), and the trace is
            // compiled afresh.
            if (CACHE.size() <= 1) {
                throw oom;
            }
            CACHE.remove(k);
            evictAllExcept(null);
            if (VERBOSE) {
                System.err.printf("[jitnn] out of device memory in %s: closed every cached plan and recompiled%n", ct.name);
            }
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse);
            CACHE.put(k, ct);
            compiles++;
            execute(ct, slots);
        }
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

    private static CompiledTrace compile(String key, List<Node> nodes, List<Storage> slots, IdentityHashMap<Storage, Integer> slotOf, int[] kinds, boolean[] written, boolean[] recyclable,
            int[] lastUse) {
        long t0 = System.nanoTime();
        String name = "jitnn" + GRAPHS.incrementAndGet();
        Object[] arrays = new Object[slots.size()];
        Storage[] bound = new Storage[slots.size()];
        IdentityHashMap<Storage, Object> byStorage = new IdentityHashMap<>();
        // Recyclable intermediates take a dead buffer of the same type and at least their size
        // (smallest that fits); it returns to the pool after its last reader. A node's outputs
        // are placed before its own inputs are released, so they never alias them.
        List<TornadoNativeArray> pool = new ArrayList<>();
        long freshBytes = 0;
        for (int ni = 0; ni < nodes.size(); ni++) {
            for (Tensor t : nodes.get(ni).outputs) {
                int si = slotOf.get(t.storage());
                if (arrays[si] != null || kinds[si] != CompiledTrace.INTERMEDIATE) {
                    continue;
                }
                Storage s = slots.get(si);
                TornadoNativeArray best = null;
                if (recyclable[si]) {
                    for (TornadoNativeArray a : pool) {
                        boolean fits = a.getSize() >= Math.max(s.numel, 1) && Storage.holds(a, s.dtype);
                        if (fits && (best == null || a.getSize() < best.getSize())) {
                            best = a;
                        }
                    }
                }
                if (best != null) {
                    pool.remove(best);
                    arrays[si] = best;
                } else {
                    arrays[si] = Storage.newArray(s.dtype, s.numel);
                    freshBytes += 4L * Math.max(s.numel, 1);
                }
            }
            for (int si = 0; si < slots.size(); si++) {
                if (recyclable[si] && lastUse[si] == ni && arrays[si] != null) {
                    pool.add((TornadoNativeArray) arrays[si]);
                }
            }
        }
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (arrays[i] == null) {
                arrays[i] = kinds[i] == CompiledTrace.PERSISTENT ? s.allocate() : Storage.newArray(s.dtype, s.numel);
            }
            if (kinds[i] == CompiledTrace.PERSISTENT) {
                bound[i] = s;
            }
            byStorage.put(s, arrays[i]);
        }
        Emitter e = new Emitter(name, byStorage);
        for (Node n : nodes) {
            e.beginNode();
            n.op.emit(e, n);
        }

        // Kind of each array the graph touches: inputs are uploaded every execution, persistent
        // storages and constants once; everything is kept on the device for later reads.
        IdentityHashMap<Object, Integer> arrayKind = new IdentityHashMap<>();
        for (Object c : e.constants) {
            arrayKind.put(c, CompiledTrace.PERSISTENT);
        }
        for (int i = 0; i < slots.size(); i++) {
            arrayKind.put(arrays[i], kinds[i]);
        }

        // WORKAROUND(TornadoVM J9): a task graph holds at most 256 tasks (and 8 KiB of
        // high-level bytecode), so long traces are cut at node boundaries into a chain of graphs
        // in one plan. Each array belongs to the first graph that uses it, which uploads and
        // persists it; later graphs consume it from the device.
        List<int[]> ranges = new ArrayList<>();
        int startNode = 0;
        int count = 0;
        int[] tasksPerNode = new int[nodes.size()];
        for (Emitter.Spec spec : e.tasks) {
            tasksPerNode[spec.node()]++;
        }
        for (int i = 0; i < nodes.size(); i++) {
            if (count > 0 && count + tasksPerNode[i] > MAX_TASKS) {
                ranges.add(new int[] { startNode, i });
                startNode = i;
                count = 0;
            }
            count += tasksPerNode[i];
        }
        ranges.add(new int[] { startNode, nodes.size() });

        IdentityHashMap<Object, String> owner = new IdentityHashMap<>();
        List<TaskGraph> graphs = new ArrayList<>();
        GridScheduler scheduler = new GridScheduler();
        int specIndex = 0;
        for (int gi = 0; gi < ranges.size(); gi++) {
            String gname = ranges.size() == 1 ? name : name + "_" + gi;
            Devices.pin(gname);
            TaskGraph graph = new TaskGraph(gname);
            List<Object> inputs = new ArrayList<>();
            List<Object> once = new ArrayList<>();
            List<Object> keep = new ArrayList<>();
            java.util.Map<String, List<Object>> consumed = new LinkedHashMap<>();
            for (int ni = ranges.get(gi)[0]; ni < ranges.get(gi)[1]; ni++) {
                for (Object a : e.uses.get(ni)) {
                    String o = owner.get(a);
                    if (o == null) {
                        owner.put(a, gname);
                        Integer k = arrayKind.get(a);
                        if (k != null && k == CompiledTrace.INPUT) {
                            inputs.add(a);
                        } else if (k != null && k == CompiledTrace.PERSISTENT) {
                            once.add(a);
                        }
                        keep.add(a);
                    } else if (!o.equals(gname)) {
                        List<Object> list = consumed.computeIfAbsent(o, x -> new ArrayList<>());
                        if (list.stream().noneMatch(x -> x == a)) {
                            list.add(a);
                        }
                    }
                }
            }
            if (!inputs.isEmpty()) {
                graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, inputs.toArray());
            }
            if (!once.isEmpty()) {
                graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, once.toArray());
            }
            for (var c : consumed.entrySet()) {
                graph.consumeFromDevice(c.getKey(), c.getValue().toArray());
            }
            while (specIndex < e.tasks.size() && e.tasks.get(specIndex).node() < ranges.get(gi)[1]) {
                Emitter.Spec spec = e.tasks.get(specIndex++);
                spec.adder().add(graph, spec.id());
                e.registerGrid(scheduler, gname, spec);
            }
            if (!keep.isEmpty()) {
                graph.persistOnDevice(keep.toArray());
            }
            graphs.add(graph);
        }

        CompiledTrace ct = new CompiledTrace(key, name, kinds, arrays, written, bound, e.tasks.size());
        ct.recyclable = recyclable;
        ct.intermediateBytes = freshBytes;
        uk.ac.manchester.tornado.api.ImmutableTaskGraph[] snapshots = new uk.ac.manchester.tornado.api.ImmutableTaskGraph[graphs.size()];
        for (int i = 0; i < snapshots.length; i++) {
            snapshots[i] = graphs.get(i).snapshot();
        }
        TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshots);
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
            System.err.printf("[jitnn] compiled %s on %s: %d nodes, %d tasks in %d graph(s), %d slots, %.1f MB of intermediates, %.1f ms%n", name,
                    plan.getDevice(0).getTornadoVMBackend() + " " + plan.getDevice(0).getDeviceName(), nodes.size(), e.tasks.size(), graphs.size(), slots.size(), freshBytes / 1e6,
                    ct.compileNanos / 1e6);
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
                // A recycled buffer holds a later tensor's data now.
                s.stale = ct.recyclable != null && ct.recyclable[i];
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

    /** Closes every cached plan except {@code keep}; false when there was nothing to close. */
    private static boolean evictAllExcept(CompiledTrace keep) {
        boolean any = false;
        var it = CACHE.values().iterator();
        while (it.hasNext()) {
            CompiledTrace ct = it.next();
            if (ct != keep) {
                close(ct);
                it.remove();
                any = true;
            }
        }
        return any;
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
