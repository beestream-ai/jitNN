package ai.beestream.jitnn.engine;

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
    /**
     * Ask TornadoVM to share kernels across array sizes (default on): a new batch or sequence size
     * then reuses the kernels compiled for earlier sizes instead of compiling every one again.
     */
    private static final boolean DYNAMIC_SHAPES = Boolean.parseBoolean(System.getProperty("jitnn.dynamicShapes", "true"));
    /**
     * Keep values a plan hands to its own next execution on the device (default on): an input that
     * is an output of the plan's previous execution (a gradient accumulated over steps) is copied
     * across on the device at the end of each execution, instead of being fetched to the host and
     * uploaded again. The plan is compiled once more when such an input is first seen.
     */
    private static final boolean CARRY = Boolean.parseBoolean(System.getProperty("jitnn.plan.carry", "true"));
    /**
     * Remember traces already run (default on): a trace recorded again with the same operators,
     * shapes, storage kinds and parameter versions replays its plan without being rewritten and
     * keyed again, which is most of jitNN's host time per call at small batches.
     */
    private static final boolean MEMO = Boolean.parseBoolean(System.getProperty("jitnn.plan.memo", "true"));
    private static final int MEMO_MAX = 64;
    private static final Map<Integer, List<Memo>> MEMOS = new java.util.HashMap<>();
    private static int memoCount;
    private static int memoStamp;
    private static long memoReplays;

    /**
     * One thread's signature of the trace it runs: the lock is released while a plan executes, so
     * threads on other devices sign their traces meanwhile.
     */
    private static final class Signed {
        long[] sig = new long[4096];
        int len;
        Op[] ops = new Op[256];
        Storage[] locals = new Storage[512];
        boolean[] produced = new boolean[512];
        long[] versions = new long[512];
        int count;
        int stamp;
    }

    private static final ThreadLocal<Signed> SIGNED = ThreadLocal.withInitial(Signed::new);

    /**
     * A trace run before: its signature, the persistent storages it read (by local number) at the
     * versions it saw, and its plan's slots as local numbers of the trace's storages, or storages
     * the rewrite made (folded weights, channels-last copies), which are internal and kept.
     */
    private record Memo(long[] sig, Op[] ops, Storage[] persistent, long[] versions, boolean[] writtenLocal, boolean[] localInSlots, String global, String key,
            CompiledTrace ct, int[] slotLocal, Storage[] slotFixed, boolean[] written) {
    }

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
            if (size() > Math.max(CACHE_SIZE, RESERVED.get()) * Math.max(1, Integer.getInteger("jitnn.concurrency", 1)) && !eldest.getValue().busy && local(eldest.getValue())) {
                close(eldest.getValue());
                return true;
            }
            return false;
        }
    };

    /**
     * Whether {@code ct} runs on this thread's device. Plans are closed only by threads of their own
     * device: another device's thread may be launching one of that device's plans meanwhile (the
     * ranks of data-parallel training run their plans at the same time).
     */
    private static boolean local(CompiledTrace ct) {
        return ct.key.startsWith(Devices.key() + "|");
    }

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
     * TornadoVM runs a plan, so threads in different slots, or on different devices, run their plans
     * at the same time. A plan that writes persistent storages marks them ({@link Storage#writer})
     * first; other threads wait on {@link #WRITTEN} before touching a marked storage.
     */
    private static final java.util.concurrent.locks.ReentrantLock LOCK = new java.util.concurrent.locks.ReentrantLock();
    private static final java.util.concurrent.locks.Condition WRITTEN = LOCK.newCondition();

    /**
     * Concurrent executions ({@code -Djitnn.concurrency}, default 1): threads are spread over this
     * many slots, each with its own compiled plans (and device buffers, weights included), so up
     * to this many threads run plans on the device at once. Threads of the same slot and device take
     * turns; threads on different devices (data-parallel ranks) do not wait for each other.
     */
    private static final int SLOTS = Math.max(1, Integer.getInteger("jitnn.concurrency", 1));
    private static final java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.locks.ReentrantLock> SLOT_LOCKS = new java.util.concurrent.ConcurrentHashMap<>();
    private static final java.util.concurrent.atomic.AtomicInteger NEXT_SLOT = new java.util.concurrent.atomic.AtomicInteger();
    private static final ThreadLocal<Integer> SLOT = ThreadLocal.withInitial(() -> NEXT_SLOT.getAndIncrement() % SLOTS);

    static void run(List<Node> recorded, Tensor[] wanted) {
        int slot = SLOT.get();
        java.util.concurrent.locks.ReentrantLock slotLock = SLOT_LOCKS.computeIfAbsent(Devices.key() + "#" + slot, k -> new java.util.concurrent.locks.ReentrantLock());
        slotLock.lock();
        try {
            LOCK.lock();
            try {
                runLocked(recorded, wanted, slot);
            } finally {
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

    /**
     * Closes this device's plans that used any of the measured {@code choices} (their next run
     * compiles with the decision) and the measurement plans (those with forced variants).
     */
    static void forgetTuning(java.util.Set<String> choices) {
        LOCK.lock();
        try {
            var it = CACHE.values().iterator();
            while (it.hasNext()) {
                CompiledTrace ct = it.next();
                boolean measured = ct.key.contains("AT{");
                boolean stale = !java.util.Collections.disjoint(ct.choices, choices);
                if ((measured || stale) && !ct.busy && local(ct)) {
                    close(ct);
                    it.remove();
                }
            }
        } finally {
            LOCK.unlock();
        }
    }

    /** Waits (releasing {@link #LOCK}) until no other thread's plan is writing {@code s}. Called with LOCK held. */
    private static void awaitWritten(Storage s, CompiledTrace self) {
        while (s.writer != null && s.writer != self) {
            WRITTEN.awaitUninterruptibly();
        }
    }

    private static void runLocked(List<Node> recorded, Tensor[] wanted, int slot) {
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
        StringBuilder key = new StringBuilder(256).append(Devices.key()).append('|').append(slotTag).append(nativeTag).append(Autotune.keyTag());
        StringBuilder ids = new StringBuilder();
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            if (!s.persistent && !produced.contains(s) && s.lease == null && ++s.inputUses >= 2 && (long) s.numel * 4 >= PROMOTE_BYTES) {
                // a large host input read again (a dataset, a constant table): keep it on the device
                // instead of copying it in on every execution. Not an array of the host pool (an
                // intermediate of the CPU backend, an output detached from its plan): the pool takes
                // it back while a plan would still hold it
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
        // Buffer reuse: an internal intermediate (a backward-pass temporary no user code can
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
        // remembered only once the rewrite has given the same plan twice: the first calls of a model
        // fold weights (batch norm, channels-last copies) that later calls take from the caches
        boolean stable = ct != null;
        if (ct == null && (ct = reuse(structure, slots, kinds)) != null) {
            CACHE.remove(ct.key);
            ct.key = k;
            CACHE.put(k, ct);
            reuses++;
        } else if (ct == null) {
            if (DEBUG_KEYS) {
                System.err.println("[jitnn] key " + k);
            }
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread, null);
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
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread, replaced.seenCarry.clone());
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
        try {
            execute(ct, slots);
        } catch (uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException oom) {
            // Device memory is shared by every cached plan. WORKAROUND(TornadoVM): a plan
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
            if (replaced != null) {
                // no longer in the cache: closed here (its carried values were read before the failure)
                close(replaced);
                replaced = null;
            }
            stable = false;
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread, null);
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
            stable = false;
            ct = compile(k, nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread, null);
            ct.structure = structure;
            BY_STRUCTURE.computeIfAbsent(structure, x -> new ArrayList<>()).add(ct);
            CACHE.put(k, ct);
            compiles++;
            execute(ct, slots);
        }
        if (replaced != null && !replaced.closed) {
            close(replaced);
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
        if (global != null && stable) {
            remember(sg, hash, global, k, ct, slots, written, recorded.size());
        }
        hostNanos += System.nanoTime() - start - (executeNanos - before);
    }

    private static void put(Signed sg, long v) {
        if (sg.len == sg.sig.length) {
            sg.sig = Arrays.copyOf(sg.sig, sg.len * 2);
        }
        sg.sig[sg.len++] = v;
    }

    /**
     * The recorded trace as numbers (its signature), its storages numbered in order of appearance
     * (locals); false when the trace is left to the full path: a large host input that may become
     * persistent (see {@link #PROMOTE_BYTES}).
     */
    private static boolean signature(Signed sg, List<Node> recorded, Tensor[] wanted) {
        int stamp = ++memoStamp;
        sg.stamp = stamp;
        sg.len = 0;
        sg.count = 0;
        if (sg.ops.length < recorded.size()) {
            sg.ops = new Op[recorded.size() * 2];
        }
        for (int i = 0; i < recorded.size(); i++) {
            Node n = recorded.get(i);
            sg.ops[i] = n.op;
            put(sg, (long) n.inputs.length << 48 | (long) n.outputs.length << 32 | (long) n.iattr.length << 16 | n.fattr.length);
            for (int v : n.iattr) {
                put(sg, v);
            }
            for (float v : n.fattr) {
                put(sg, Float.floatToRawIntBits(v));
            }
            for (Tensor t : n.inputs) {
                signTensor(sg, t, false);
            }
            for (Tensor t : n.outputs) {
                signTensor(sg, t, true);
            }
        }
        put(sg, -3);
        for (Tensor t : wanted) {
            Storage s = t.storage();
            put(sg, s.memoStamp == stamp ? s.memoLocal : -1);
        }
        for (int i = 0; i < sg.count; i++) {
            Storage s = sg.locals[i];
            if (!s.persistent && !sg.produced[i] && s.lease == null && (long) s.numel * 4 >= PROMOTE_BYTES) {
                return false;
            }
        }
        return true;
    }

    private static void signTensor(Signed sg, Tensor t, boolean output) {
        if (t == null) {
            put(sg, -2);
            return;
        }
        Storage s = t.storage();
        if (s.memoStamp != sg.stamp) {
            if (sg.count == sg.locals.length) {
                sg.locals = Arrays.copyOf(sg.locals, sg.count * 2);
                sg.produced = Arrays.copyOf(sg.produced, sg.count * 2);
                sg.versions = Arrays.copyOf(sg.versions, sg.count * 2);
            }
            s.memoStamp = sg.stamp;
            s.memoLocal = sg.count;
            sg.locals[sg.count] = s;
            sg.versions[sg.count] = s.version;
            sg.produced[sg.count++] = false;
        }
        if (output) {
            sg.produced[s.memoLocal] = true;
        }
        ai.beestream.jitnn.Shape shape = t.shape();
        int flags = s.dtype.ordinal() | (s.persistent ? 1 << 8 : 0) | (s.internal ? 1 << 9 : 0) | (s.lease != null ? 1 << 10 : 0) | (s.released ? 1 << 11 : 0)
                | (t.requiresGrad() ? 1 << 12 : 0);
        put(sg, (long) s.memoLocal << 32 | flags);
        put(sg, (long) s.numel << 8 | shape.rank());
        for (int d = 0; d < shape.rank(); d++) {
            put(sg, shape.dim(d));
        }
    }

    private static int memoHash(Signed sg) {
        int h = sg.count;
        for (int i = 0; i < sg.len; i++) {
            h = 31 * h + Long.hashCode(sg.sig[i]);
        }
        return h;
    }

    /** Runs the remembered plan of the trace just signed, if there is one that still holds. */
    private static boolean replay(Signed sg, int hash, String global, List<Node> recorded, Tensor[] wanted) {
        List<Memo> list = MEMOS.get(hash);
        if (list == null) {
            return false;
        }
        for (Memo m : list) {
            CompiledTrace ct = m.ct();
            if (!matches(sg, m, global, recorded.size()) || ct.closed || CACHE.get(m.key()) != ct || ct.seenCarry != null && ct.carry == null) {
                continue;
            }
            List<Storage> slots = new ArrayList<>(m.slotLocal().length);
            for (int i = 0; i < m.slotLocal().length; i++) {
                slots.add(m.slotLocal()[i] >= 0 ? sg.locals[m.slotLocal()[i]] : m.slotFixed()[i]);
            }
            // the outputs to fetch, decided now: storages' numbers belong to whichever thread signed last
            Storage[] fetches = new Storage[wanted.length];
            for (int i = 0; i < wanted.length; i++) {
                Storage s = wanted[i].storage();
                fetches[i] = s.memoStamp == sg.stamp && m.localInSlots()[s.memoLocal] ? s : null;
            }
            replays++;
            memoReplays++;
            ct.lastUsed = ++executionsTotal;
            lastIntermediateBytes = ct.intermediateBytes;
            evictIdle(ct);
            try {
                execute(ct, slots);
            } catch (RuntimeException failure) {
                // out of device memory, or a library failing: the full path handles it
                MEMOS.clear();
                memoCount = 0;
                return false;
            }
            for (Node n : recorded) {
                n.executed = true;
            }
            for (int i = 0; i < slots.size(); i++) {
                if (m.written()[i]) {
                    slots.get(i).version++;
                }
            }
            for (Storage s : fetches) {
                if (s != null && !s.hostValid) {
                    fetch(s);
                }
            }
            return true;
        }
        return false;
    }

    private static boolean matches(Signed sg, Memo m, String global, int nodes) {
        if (m.sig().length != sg.len || m.ops().length != nodes || m.persistent().length != sg.count || !m.global().equals(global)) {
            return false;
        }
        for (int i = 0; i < nodes; i++) {
            if (m.ops()[i] != sg.ops[i]) {
                return false;
            }
        }
        if (!Arrays.equals(m.sig(), 0, sg.len, sg.sig, 0, sg.len)) {
            return false;
        }
        for (int i = 0; i < sg.count; i++) {
            Storage p = m.persistent()[i];
            // the same persistent storages, and those the trace only reads unchanged (folded and
            // channels-last weights were made from them)
            if (p != null && (p != sg.locals[i] || !m.writtenLocal()[i] && p.version != m.versions()[i])) {
                return false;
            }
        }
        return true;
    }

    /** Remembers the plan a trace just ran (its storages numbered by {@link #signature}). */
    private static void remember(Signed sg, int hash, String global, String key, CompiledTrace ct, List<Storage> slots, boolean[] written, int nodes) {
        IdentityHashMap<Storage, Integer> local = new IdentityHashMap<>();
        for (int i = 0; i < sg.count; i++) {
            local.put(sg.locals[i], i);
        }
        int[] slotLocal = new int[slots.size()];
        Storage[] slotFixed = new Storage[slots.size()];
        boolean[] localInSlots = new boolean[sg.count];
        for (int i = 0; i < slots.size(); i++) {
            Storage s = slots.get(i);
            Integer l = local.get(s);
            if (l != null) {
                slotLocal[i] = l;
                localInSlots[l] = true;
            } else {
                // made by the rewrite (a folded weight, a channels-last copy): no user code holds it
                slotLocal[i] = -1;
                slotFixed[i] = s;
            }
        }
        Storage[] persistent = new Storage[sg.count];
        long[] versions = new long[sg.count];
        boolean[] writtenLocal = Arrays.copyOf(sg.produced, sg.count);
        for (int i = 0; i < sg.count; i++) {
            if (sg.locals[i].persistent) {
                persistent[i] = sg.locals[i];
                // the version the trace was rewritten for
                versions[i] = sg.versions[i];
            }
        }
        if (memoCount >= MEMO_MAX) {
            MEMOS.clear();
            memoCount = 0;
        }
        MEMOS.computeIfAbsent(hash, h -> new ArrayList<>()).add(new Memo(Arrays.copyOf(sg.sig, sg.len), Arrays.copyOf(sg.ops, nodes), persistent, versions, writtenLocal,
                localInSlots, global, key, ct, slotLocal, slotFixed, written.clone()));
        memoCount++;
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

    /**
     * Drops a plan from the structure index and from the replay memos (it was closed or dropped from
     * the cache). A memo would otherwise keep the closed plan, its host arrays and the storages it was
     * bound to reachable until 64 more traces were remembered.
     */
    private static void forget(CompiledTrace ct) {
        ct.closed = true;
        List<CompiledTrace> list = ct.structure == null ? null : BY_STRUCTURE.get(ct.structure);
        if (list != null) {
            list.remove(ct);
            if (list.isEmpty()) {
                BY_STRUCTURE.remove(ct.structure);
            }
        }
        var memos = MEMOS.values().iterator();
        while (memos.hasNext()) {
            List<Memo> l = memos.next();
            int before = l.size();
            l.removeIf(m -> m.ct() == ct);
            memoCount -= before - l.size();
            if (l.isEmpty()) {
                memos.remove();
            }
        }
    }

    private static int add(List<Storage> slots, Storage s) {
        slots.add(s);
        return slots.size() - 1;
    }

    private static CompiledTrace compile(String key, List<Node> nodes, List<Storage> slots, IdentityHashMap<Storage, Integer> slotOf, int[] kinds, boolean[] written, boolean[] recyclable,
            int[] lastUse, Set<Storage> unread, int[] carry) {
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
            e.beginNode(n);
            n.op.emit(e, n);
        }
        // Carried values: after the last node, each carried intermediate is copied into the input
        // buffer the next execution reads, which is then uploaded from the host only when needed.
        Set<Object> carried = Collections.newSetFromMap(new IdentityHashMap<>());
        if (carry != null) {
            for (int i = 0; i < carry.length; i++) {
                int j = carry[i];
                boolean ok = j >= 0 && kinds[i] == CompiledTrace.INPUT && kinds[j] == CompiledTrace.INTERMEDIATE && !recyclable[j] && !unread.contains(slots.get(j))
                        && arrays[i] instanceof FloatArray && arrays[j] instanceof FloatArray && slots.get(i).numel == slots.get(j).numel;
                if (ok) {
                    e.carry((FloatArray) arrays[j], (FloatArray) arrays[i], slots.get(i).numel);
                    carried.add(arrays[i]);
                } else {
                    carry[i] = -1;
                }
            }
        }

        // Kind of each array the graph touches: inputs are uploaded every execution (carried ones
        // once, then on demand), persistent storages and constants once; everything is kept on the
        // device for later reads.
        IdentityHashMap<Object, Integer> arrayKind = new IdentityHashMap<>();
        for (Object c : e.constants) {
            arrayKind.put(c, CompiledTrace.PERSISTENT);
        }
        for (int i = 0; i < slots.size(); i++) {
            arrayKind.put(arrays[i], kinds[i]);
        }

        // WORKAROUND(TornadoVM): a task graph holds at most 256 tasks (and 8 KiB of
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
                        if (k != null && k == CompiledTrace.INPUT && carried.contains(a)) {
                            once.add(a);
                        } else if (k != null && k == CompiledTrace.INPUT) {
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
            java.util.Set<String> forked = new java.util.LinkedHashSet<>();
            while (specIndex < e.tasks.size() && e.tasks.get(specIndex).node() < ranges.get(gi)[1]) {
                Emitter.Spec spec = e.tasks.get(specIndex++);
                if (spec.joinKind() == 2 && CUDA_GRAPH && !forked.remove(spec.join())) {
                    // its fork was in an earlier task graph, which joined it at its end
                    continue;
                }
                if (spec.joinKind() == 1) {
                    forked.add(spec.join());
                }
                spec.adder().add(graph, spec.id());
                e.registerGrid(scheduler, gname, spec);
                if (labels != null) {
                    Node nd = nodes.get(spec.node());
                    String shape = nd.op.name().equals("matmul") || nd.op.name().equals("linear") || nd.op.name().startsWith("conv2d") ? java.util.Arrays.toString(nd.iattr) : "";
                    labels.put(gname + "." + spec.id(), nd.op.name() + shape + "#" + perNode[spec.node()]++);
                }
            }
            if (CUDA_GRAPH) {
                // a captured graph may not end with a side stream forked: join what is still open
                int k = 0;
                for (String open : forked) {
                    e.joins.get(open).add(graph, "join" + k++);
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
        ct.carry = carry;
        ct.libraries = e.libraries;
        for (Emitter.Spec spec : e.tasks) {
            if (spec.grid() == null) {
                ct.libraryCalls++;
            } else {
                ct.kernels++;
            }
        }
        for (Node n : nodes) {
            ct.tensorBytes += tensorBytes(n);
        }
        ct.choices = e.choices;
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
        if (DYNAMIC_SHAPES) {
            plan.withDynamicShapes();
        }
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

    /** The plan this thread executed last (Autotune times it again). */
    private static final ThreadLocal<CompiledTrace> LAST = new ThreadLocal<>();

    /**
     * Runs the plan this thread executed last {@code runs} more times, as it is (no recording, no
     * lookup), and returns the median seconds per execution: the device time of its kernels plus
     * the plan's launch. For Autotune, on its measurement plans.
     */
    static double timeLast(int runs) {
        CompiledTrace ct = LAST.get();
        double[] t = new double[runs];
        for (int r = -2; r < runs; r++) {
            long t0 = System.nanoTime();
            ct.plan.execute();
            if (r >= 0) {
                t[r] = (System.nanoTime() - t0) / 1e9;
            }
        }
        java.util.Arrays.sort(t);
        return t[runs / 2];
    }

    private static void execute(CompiledTrace ct, List<Storage> slots) {
        LAST.set(ct);
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
                    WRITTEN.signalAll();
                }
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
            lastKernelNanos = ct.last.getProfilerResult().getDeviceKernelTime();
            lastTotalNanos = ct.last.getProfilerResult().getTotalTime();
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

    /**
     * The intermediate slot of {@code ct} that holds {@code s} from the plan's previous execution
     * (its buffer still has the value on the device), or -1.
     */
    private static int previousOutput(CompiledTrace ct, Storage s) {
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

    private static int indexOf(CompiledTrace ct, Storage s) {
        for (int i = 0; i < ct.bound.length; i++) {
            if (ct.bound[i] == s) {
                return i;
            }
        }
        return 0;
    }

    /** Inputs from this size are copied into a plan's buffer by several threads. */
    private static final long PARALLEL_COPY_BYTES = Long.getLong("jitnn.parallelCopyBytes", 32L << 20);
    private static final int COPY_THREADS = Math.min(8, Runtime.getRuntime().availableProcessors());

    /**
     * Copies an input into a plan's buffer. One thread copied ResNet-50's [128, 3, 224, 224] FP32
     * image (77 MB) in 6.3 ms, 8% of the call; large inputs are split across the common pool.
     */
    private static void stage(MemorySegment from, MemorySegment to, long bytes) {
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
                // an output not read yet: keep its data on the host, as the plan's buffers go. The
                // plan's host array becomes the storage's own, unless another slot of the plan shares it
                ct.last.transferToHost(s.array);
                if (sharedArray(ct, s.array)) {
                    TornadoNativeArray own = Storage.newArray(s.dtype, s.numel);
                    copy((TornadoNativeArray) s.array, own);
                    s.array = own;
                }
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

    /** Whether more than one slot of {@code ct} uses {@code array} (a recycled buffer). */
    private static boolean sharedArray(CompiledTrace ct, Object array) {
        int uses = 0;
        for (Object a : ct.arrays) {
            if (a == array && ++uses > 1) {
                return true;
            }
        }
        return false;
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
            if (ct != running && !ct.busy && local(ct) && ct.intermediateBytes >= IDLE_BYTES && executionsTotal - ct.lastUsed > IDLE_EXECUTIONS) {
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
            if (ct != keep && !ct.busy && local(ct)) {
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
            MEMOS.clear();
            memoCount = 0;
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

    // ---------------------------------------------------------------- timeline (Chrome trace events)

    private static final java.util.regex.Pattern TASK_ID = java.util.regex.Pattern.compile("(.*?)(?:_(\\d+))?\\.t(\\d+)");
    private static List<String> TRACE;
    private static long traceOrigin;

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
    private static void traceExecution(CompiledTrace ct, String log, long start, long end) {
        java.util.Map<String, long[]> now = new java.util.HashMap<>();
        java.util.regex.Matcher m = TASK_TIME.matcher(log == null ? "" : log);
        while (m.find()) {
            long[] acc = now.computeIfAbsent(m.group(1), k -> new long[2]);
            acc[0]++;
            acc[1] += Long.parseLong(m.group(2));
        }
        List<String> tasks = new ArrayList<>(now.keySet());
        tasks.sort(java.util.Comparator.comparingLong(TornadoExecutor::taskOrder));
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
    private static long taskOrder(String task) {
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

    private static long bytes(Tensor t) {
        ai.beestream.jitnn.DType d = t.dtype();
        long numel = t.numel();
        return d.isBlockQuantized() ? numel / d.blockValues() * d.blockBytes() : numel * d.bytes();
    }

    /** Per plan execution so far: {@code [executions, kernels, library calls, tensor bytes]}. */
    private static final long[] TRAFFIC = new long[4];

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

    private static long lastIntermediateBytes;

    /** Device bytes of intermediates (buffers other than inputs and persistent storages) of the last plan run. */
    static long lastIntermediateBytes() {
        LOCK.lock();
        try {
            return lastIntermediateBytes;
        } finally {
            LOCK.unlock();
        }
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
