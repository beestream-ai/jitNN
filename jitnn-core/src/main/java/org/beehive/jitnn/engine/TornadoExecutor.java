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
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
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
 * compiling one. A plan whose persistent storages were all released (their model was closed) is
 * reused by a trace of the same structure over other persistent storages, such as the same model
 * loaded again: their contents are copied into the plan's buffers and uploaded, and nothing is
 * compiled.
 */
final class TornadoExecutor {

    private static final int CACHE_SIZE = Integer.getInteger("jitnn.plan.cache", 16);
    private static final boolean VERBOSE = Boolean.getBoolean("jitnn.verbose");
    /** Host inputs at least this large that are read again become persistent device storage. */
    private static final long PROMOTE_BYTES = Long.getLong("jitnn.promoteInputBytes", 16L << 20);
    /** Prints the cache key of every compiled plan (to find what makes a trace recompile). */
    private static final boolean DEBUG_KEYS = Boolean.getBoolean("jitnn.debugKeys");
    private static final boolean PROFILE = Boolean.getBoolean("jitnn.profile");
    /**
     * Tasks per TornadoVM task graph before the trace is split into a chain of graphs. A graph's
     * high-level bytecode buffer is 8 KiB and a task takes 15 + 5 bytes per argument (jitNN
     * kernels take up to 19), next to the transfer declarations: 40 tasks stays inside it.
     */
    private static final int MAX_TASKS = Integer.getInteger("jitnn.graph.maxTasks", 40);
    /** Reuse dead intermediates' buffers within a compiled trace (default on). */
    private static final boolean REUSE = Boolean.parseBoolean(System.getProperty("jitnn.reuseBuffers", "true"));
    /**
     * Reuse only buffers of exactly an intermediate's size ({@code -Djitnn.reuseBuffers.exact}): {@code auto}
     * (the default) on Metal with MLX, {@code true} always, {@code false} never.
     */
    private static final String EXACT_REUSE = System.getProperty("jitnn.reuseBuffers.exact", "auto");
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
            // every concurrency slot compiles its own plans, so the cache holds CACHE_SIZE per slot
            if (size() > Math.max(CACHE_SIZE, RESERVED.get()) * Math.max(1, Integer.getInteger("jitnn.concurrency", 1)) && !eldest.getValue().busy) {
                close(eldest.getValue());
                return true;
            }
            return false;
        }
    };

    /** Plan cache room reserved by users that need many plans at once (Engine.reservePlans: shape buckets). */
    static final java.util.concurrent.atomic.AtomicInteger RESERVED = new java.util.concurrent.atomic.AtomicInteger();

    /** Cached plans by structure (the key without persistent storage identities), for plan reuse. */
    private static final Map<String, List<CompiledTrace>> BY_STRUCTURE = new java.util.HashMap<>();

    private static long compiles;
    private static long replays;
    private static long reuses;

    private TornadoExecutor() {
    }

    static boolean isLoaded() {
        return loaded;
    }

    /**
     * The executor's lock: plan cache, storage bookkeeping and counters. It is released while
     * TornadoVM runs a plan that writes no persistent storage, so threads in different slots run
     * their plans at the same time.
     */
    private static final java.util.concurrent.locks.ReentrantLock LOCK = new java.util.concurrent.locks.ReentrantLock();

    /**
     * Concurrent executions ({@code -Djitnn.concurrency}, default 1): threads are spread over this
     * many slots, each with its own compiled plans (and device buffers, weights included), so up
     * to this many threads run plans on the device at once. Threads of the same slot take turns.
     */
    private static final int SLOTS = Math.max(1, Integer.getInteger("jitnn.concurrency", 1));
    private static final java.util.concurrent.locks.ReentrantLock[] SLOT_LOCKS = new java.util.concurrent.locks.ReentrantLock[SLOTS];
    private static final java.util.concurrent.atomic.AtomicInteger NEXT_SLOT = new java.util.concurrent.atomic.AtomicInteger();
    private static final ThreadLocal<Integer> SLOT = ThreadLocal.withInitial(() -> NEXT_SLOT.getAndIncrement() % SLOTS);

    static {
        for (int i = 0; i < SLOTS; i++) {
            SLOT_LOCKS[i] = new java.util.concurrent.locks.ReentrantLock();
        }
    }

    static void run(List<Node> recorded, Tensor[] wanted) {
        int slot = SLOT.get();
        SLOT_LOCKS[slot].lock();
        try {
            LOCK.lock();
            try {
                runLocked(recorded, wanted, slot);
            } finally {
                LOCK.unlock();
            }
        } finally {
            SLOT_LOCKS[slot].unlock();
        }
    }

    private static void runLocked(List<Node> recorded, Tensor[] wanted, int slot) {
        long start = System.nanoTime();
        long before = executeNanos;
        loaded = true;
        List<Node> nodes = org.beehive.jitnn.ops.Fusion.rewrite(recorded, wanted);
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
        String nativeTag = Native.keyTag();
        String slotTag = SLOTS > 1 ? "S" + slot + "|" : "";
        StringBuilder key = new StringBuilder(256).append(Devices.key()).append('|').append(slotTag).append(nativeTag);
        StringBuilder ids = new StringBuilder();
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (!s.persistent && !produced.contains(s) && ++s.inputUses >= 2 && (long) s.numel * 4 >= PROMOTE_BYTES) {
                // a large host input read again (a dataset, a constant table): keep it on the device
                // instead of copying it in on every execution
                Engine.ensureHost(s);
                s.persistent = true;
                s.internal = false;
            }
            if (s.persistent) {
                if (s.released) {
                    throw new IllegalStateException("a tensor of a closed model was used (persistent storage " + s.id + " was released)");
                }
                kinds[i] = CompiledTrace.PERSISTENT;
                written[i] = produced.contains(s);
                key.append('P').append(written[i] ? 'w' : 'r');
                ids.append(s.id).append(',');
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
        // outputs nothing reads: internal (no user code holds them), not consumed in the trace, not
        // requested; an operator may skip them (a library call without the saved statistics)
        Set<Storage> unread = Collections.newSetFromMap(new IdentityHashMap<>());
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (kinds[i] == CompiledTrace.INTERMEDIATE && s.internal && !consumed[i] && !wantedSet.contains(s)) {
                unread.add(s);
            }
        }
        String structure = key.toString();
        String k = structure + "|P" + ids;
        CompiledTrace ct = CACHE.get(k);
        if (ct == null && (ct = reuse(structure, slots, kinds)) != null) {
            CACHE.remove(ct.key);
            ct.key = k;
            CACHE.put(k, ct);
            reuses++;
        } else if (ct == null) {
            if (DEBUG_KEYS) {
                System.err.println("[jitnn] key " + k);
            }
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread);
            ct.structure = structure;
            BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
            CACHE.put(k, ct);
            compiles++;
        } else {
            replays++;
        }
        ct.lastUsed = ++executionsTotal;
        evictIdle(ct);
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
            forget(ct);
            evictAllExcept(null);
            if (VERBOSE) {
                System.err.printf("[jitnn] out of device memory in %s: closed every cached plan and recompiled%n", ct.name);
            }
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread);
            ct.structure = structure;
            BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
            CACHE.put(k, ct);
            compiles++;
            execute(ct, slots);
        } catch (RuntimeException failure) {
            if (ct.executions > 0 || ct.libraries.isEmpty()) {
                throw failure;
            }
            // a vendor library failed on its first call (missing at run time, or an argument form it
            // does not take): stop using it and compile the trace with jitNN's kernels
            CACHE.remove(k);
            forget(ct);
            closeQuietly(ct);
            Native.disable(ct.libraries, failure);
            String tag = Native.keyTag();
            structure = Devices.key() + "|" + slotTag + tag + structure.substring(Devices.key().length() + 1 + slotTag.length() + nativeTag.length());
            k = structure + "|P" + ids;
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread);
            ct.structure = structure;
            BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
            CACHE.put(k, ct);
            compiles++;
            execute(ct, slots);
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
            if (!s.hostValid && slotOf.containsKey(s)) {
                fetch(s);
            }
        }
        hostNanos += System.nanoTime() - start - (executeNanos - before);
    }

    /**
     * A cached plan of this structure whose persistent storages are those of the trace or were
     * released, rebound to the trace's storages; null if there is none. Each rebound storage takes
     * over the plan's buffer (its host contents are copied in, and uploaded at the next execution).
     */
    private static CompiledTrace reuse(String structure, List<Storage> slots, int[] kinds) {
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

    /** The sum of every {@code "field": "n"} in TornadoVM's profiler log (nanoseconds). */
    private static long sumField(String log, String field) {
        long sum = 0;
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("\"" + field + "\"\\s*:\\s*\"?(\\d+)").matcher(log == null ? "" : log);
        while (m.find()) {
            sum += Long.parseLong(m.group(1));
        }
        return sum;
    }

    /** Releases a plan that failed before it ran (nothing of it is on the host to save). */
    private static void closeQuietly(CompiledTrace ct) {
        try {
            ct.plan.close();
        } catch (TornadoExecutionPlanException | RuntimeException ignored) {
            // the plan is unusable either way
        }
    }

    /** Drops a plan from the structure index (it was closed or dropped from the cache). */
    private static void forget(CompiledTrace ct) {
        ct.closed = true;
        List<CompiledTrace> list = ct.structure == null ? null : BY_STRUCTURE.get(ct.structure);
        if (list != null) {
            list.remove(ct);
            if (list.isEmpty()) {
                BY_STRUCTURE.remove(ct.structure);
            }
        }
    }

    private static int add(List<Storage> slots, Storage s) {
        slots.add(s);
        return slots.size() - 1;
    }

    private static CompiledTrace compile(String key, List<Node> nodes, List<Storage> slots, IdentityHashMap<Storage, Integer> slotOf, int[] kinds, boolean[] written, boolean[] recyclable,
            int[] lastUse, Set<Storage> unread) {
        long t0 = System.nanoTime();
        String name = "jitnn" + GRAPHS.incrementAndGet();
        Object[] arrays = new Object[slots.size()];
        Storage[] bound = new Storage[slots.size()];
        IdentityHashMap<Storage, Object> byStorage = new IdentityHashMap<>();
        // Recyclable intermediates take a dead buffer of the same type and at least their size
        // (smallest that fits); it returns to the pool after its last reader. A node's outputs
        // are placed before its own inputs are released, so they never alias them.
        List<TornadoNativeArray> pool = new ArrayList<>();
        // with MLX on Metal only buffers of exactly the size are reused: MLX takes only arrays of exactly
        // its operands' size, and a larger reused buffer costs a copy in and out of every library call
        boolean exact = EXACT_REUSE.equals("true") || EXACT_REUSE.equals("auto") && Devices.backend() == uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.METAL
                && Native.use("gemm", Native.Lib.MLX);
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
                        boolean fits = (exact ? a.getSize() == Math.max(s.numel, 1) : a.getSize() >= Math.max(s.numel, 1)) && Storage.holds(a, s.dtype);
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
                s.planBound = true;
            }
            byStorage.put(s, arrays[i]);
        }
        Emitter e = new Emitter(name, byStorage, unread);
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
        java.util.Map<String, String> labels = PROFILE ? new java.util.HashMap<>() : null;
        int[] perNode = PROFILE ? new int[nodes.size()] : null;
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
                if (labels != null) {
                    Node nd = nodes.get(spec.node());
                    String shape = nd.op.name().equals("matmul") || nd.op.name().equals("linear") || nd.op.name().startsWith("conv2d") ? java.util.Arrays.toString(nd.iattr) : "";
                    labels.put(gname + "." + spec.id(), nd.op.name() + shape + "#" + perNode[spec.node()]++);
                }
            }
            if (!keep.isEmpty()) {
                graph.persistOnDevice(keep.toArray());
            }
            graphs.add(graph);
        }

        CompiledTrace ct = new CompiledTrace(key, name, kinds, arrays, written, bound, e.tasks.size());
        ct.deviceArrays = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        ct.deviceArrays.addAll(owner.keySet());
        ct.recyclable = recyclable;
        ct.libraries = e.libraries;
        ct.taskLabels = labels;
        ct.intermediateBytes = freshBytes;
        uk.ac.manchester.tornado.api.ImmutableTaskGraph[] snapshots = new uk.ac.manchester.tornado.api.ImmutableTaskGraph[graphs.size()];
        for (int i = 0; i < snapshots.length; i++) {
            snapshots[i] = graphs.get(i).snapshot();
        }
        TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshots);
        if (e.tasks.stream().anyMatch(t -> t.grid() != null)) {
            // a plan of library tasks only has no grids, and TornadoVM rejects an empty scheduler
            plan.withGridScheduler(scheduler);
        }
        if (PROFILE) {
            plan.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.SILENT);
        }
        Devices.apply(plan);
        if (CUDA_GRAPH && Devices.backend() == uk.ac.manchester.tornado.api.enums.TornadoVMBackendType.CUDA) {
            plan.withCUDAGraph();
        }
        if (Engine.kernelSink() != null) {
            plan.withPrintKernel();
            ct.printHeader = describe(name, plan, nodes, tasksPerNode, graphs.size());
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

    /** The operators of a plan, for kernel printing. */
    private static String describe(String name, TornadoExecutionPlan plan, List<Node> nodes, int[] tasksPerNode, int graphs) {
        StringBuilder b = new StringBuilder();
        b.append("// ==== jitNN plan ").append(name).append(" on ").append(plan.getDevice(0).getTornadoVMBackend()).append(' ').append(plan.getDevice(0).getDeviceName())
                .append(": ").append(nodes.size()).append(" operators, ").append(Arrays.stream(tasksPerNode).sum()).append(" kernels in ").append(graphs).append(" graph(s)\n");
        int task = 0;
        for (int i = 0; i < nodes.size(); i++) {
            Node n = nodes.get(i);
            b.append("//   [").append(i).append("] ").append(n.op.name());
            if (n.iattr != null && n.iattr.length > 0) {
                b.append(' ').append(Arrays.toString(n.iattr));
            }
            b.append(' ');
            for (int k = 0; k < n.inputs.length; k++) {
                b.append(k == 0 ? "" : ", ").append(n.inputs[k].shape()).append(' ').append(n.inputs[k].dtype());
            }
            b.append(" -> ");
            for (int k = 0; k < n.outputs.length; k++) {
                b.append(k == 0 ? "" : ", ").append(n.outputs[k].shape()).append(' ').append(n.outputs[k].dtype());
            }
            if (tasksPerNode[i] == 0) {
                b.append("  (no kernel)");
            } else {
                b.append("  tasks t").append(task);
                if (tasksPerNode[i] > 1) {
                    b.append("..t").append(task + tasksPerNode[i] - 1);
                }
            }
            task += tasksPerNode[i];
            b.append('\n');
        }
        return b.toString();
    }

    private static final Object PRINT_LOCK = new Object();

    /** The first run of a plan with kernel printing on: collects TornadoVM's source dump from standard output. */
    private static TornadoExecutionResult executePrinting(CompiledTrace ct) {
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

    private static void execute(CompiledTrace ct, List<Storage> slots) {
        // Inputs first: an input may be an intermediate of this plan's previous execution, which
        // must be fetched before its slot is rebound below.
        for (int i = 0; i < slots.size(); i++) {
            if (ct.kinds[i] == CompiledTrace.INPUT) {
                Storage s = slots.get(i);
                Engine.ensureHost(s);
                transfers[4]++;
                transfers[5] += s.numel;
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
        boolean writes = false;
        for (boolean w : ct.written) {
            writes |= w;
        }
        if (writes || ct.printHeader != null) {
            // persistent storages are written: other plans must not run on them meanwhile
            ct.last = ct.printHeader != null ? executePrinting(ct) : ct.plan.execute();
        } else {
            ct.busy = true;
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
            }
        }
        executeNanos += System.nanoTime() - t0;
        if (VERBOSE && ct.executions == 0) {
            String split = "";
            if (PROFILE) {
                // TornadoVM's per-task compile times: Graal (Java to PTX/OpenCL C) and the driver (to binary)
                String log = ct.last.getProfilerResult().getProfileLog();
                split = String.format(" (Graal %.1f ms, driver %.1f ms)", sumField(log, "TASK_COMPILE_GRAAL_TIME") / 1e6, sumField(log, "TASK_COMPILE_DRIVER_TIME") / 1e6);
            }
            System.err.printf("[jitnn] first execution of %s (kernel compilation): %.1f ms%s%n", ct.name, (System.nanoTime() - t0) / 1e6, split);
        }
        if (ct.executions == 0 && !ct.libraries.isEmpty()) {
            Native.proven(ct.libraries);
        }
        ct.executions++;
        if (PROFILE) {
            lastKernelNanos = ct.last.getProfilerResult().getDeviceKernelTime();
            lastTotalNanos = ct.last.getProfilerResult().getTotalTime();
            accumulate(ct, ct.last.getProfilerResult().getProfileLog());
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
    static void fetch(Storage s) {
        LOCK.lock();
        try {
            if (s.hostValid) {
                return;
            }
            transfers[0]++;
            transfers[1] += s.numel;
            if (s.persistent) {
                if (s.validOn.isEmpty()) {
                    throw new IllegalStateException("persistent storage " + s.id + " has no valid copy");
                }
                CompiledTrace ct = s.validOn.iterator().next();
                for (CompiledTrace c : s.validOn) {
                    // a plan another thread is running is not read from
                    if (!c.busy) {
                        ct = c;
                        break;
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
                // an output not read yet: keep its data on the host, as the plan's buffers go
                ct.last.transferToHost(s.array);
                TornadoNativeArray own = Storage.newArray(s.dtype, s.numel);
                copy((TornadoNativeArray) s.array, own);
                s.array = own;
                s.hostValid = true;
                s.boundTrace = null;
            }
        }
        try {
            ct.plan.close();
        } catch (TornadoExecutionPlanException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Executions of any plan so far; a plan's lastUsed is this count at its latest execution. */
    private static long executionsTotal;

    /**
     * Plans holding at least this many bytes of intermediates are closed once IDLE_EXECUTIONS
     * executions of other plans pass without them (a first step that is never repeated, a
     * validation pass between epochs), freeing their device buffers. A closed plan that comes back
     * is compiled again.
     */
    private static final long IDLE_BYTES = Long.getLong("jitnn.plan.idleBytes", 64L << 20);
    private static final long IDLE_EXECUTIONS = Long.getLong("jitnn.plan.idleExecutions", 256);

    private static void evictIdle(CompiledTrace running) {
        if ((executionsTotal & 31) != 0) {
            return;
        }
        var it = CACHE.values().iterator();
        while (it.hasNext()) {
            CompiledTrace ct = it.next();
            if (ct != running && !ct.busy && ct.intermediateBytes >= IDLE_BYTES && executionsTotal - ct.lastUsed > IDLE_EXECUTIONS) {
                if (VERBOSE) {
                    System.err.printf("[jitnn] closed idle plan %s (%.1f MB of intermediates)%n", ct.name, ct.intermediateBytes / 1e6);
                }
                close(ct);
                it.remove();
            }
        }
    }

    /** Closes every cached plan except {@code keep}; false when there was nothing to close. */
    private static boolean evictAllExcept(CompiledTrace keep) {
        boolean any = false;
        var it = CACHE.values().iterator();
        while (it.hasNext()) {
            CompiledTrace ct = it.next();
            if (ct != keep && !ct.busy) {
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
            for (CompiledTrace ct : CACHE.values()) {
                close(ct);
            }
            CACHE.clear();
        } finally {
            LOCK.unlock();
        }
    }

    private static final java.util.regex.Pattern TASK_TIME = java.util.regex.Pattern.compile(
            "\"([A-Za-z0-9_]+\\.t\\d+)\"\\s*:\\s*\\{[^}]*?\"TASK_KERNEL_TIME\"\\s*:\\s*\"(\\d+)\"");
    private static final java.util.Map<String, long[]> BREAKDOWN = new java.util.TreeMap<>();

    /** Adds the per-task kernel times of one profiled execution to the per-operator totals. */
    private static void accumulate(CompiledTrace ct, String log) {
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

    /** With -Djitnn.profile=true: [device kernel ns, TornadoVM total ns] of the last execution. */
    static long[] lastProfile() {
        LOCK.lock();
        try {
            return new long[] { lastKernelNanos, lastTotalNanos };
        } finally {
            LOCK.unlock();
        }
    }

    /**
     * Host-device traffic the executor causes: [persistent or output fetches to the host,
     * their elements, persistent re-uploads (a storage written by another plan), their elements,
     * input copies, their elements].
     */
    private static final long[] transfers = new long[6];

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

    /** Compile and replay counts, for tests and benchmarks. */
    static long[] stats() {
        LOCK.lock();
        try {
            return new long[] { compiles, replays, reuses };
        } finally {
            LOCK.unlock();
        }
    }
}
