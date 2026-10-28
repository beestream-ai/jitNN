package org.beehive.jitnn.engine;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

import org.beehive.jitnn.Tensor;

import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * Runs the operators recorded since the last sync on the CPU backend: the trace is rewritten by the
 * same fusion pass as on a device ({@code Fusion.rewrite}), then each node runs in program order with
 * its operator's CPU kernel ({@link Op#cpu}) or, when it has none, its host reference. Output arrays
 * come from the host pool; an internal intermediate (no user code holds it) that is not requested
 * goes back to the pool once its last reader has run, so a step's temporaries reuse each other's
 * memory as on a device.
 *
 * <p>Calls and reference fallbacks are counted per operator ({@link Engine#cpuStats});
 * {@code -Djitnn.cpu.stats=true} also times them, and the host work before each operator (fetching
 * inputs, taking output arrays) as "(outputs)", and prints the table at exit.
 *
 * <p>The host pool keeps up to {@code jitnn.cpu.poolBytes} (default 2 GiB) of arrays on this
 * backend, unless {@code jitnn.hostPool.bytes} is set.
 */
final class CpuExecutor {

    static final boolean STATS = Boolean.getBoolean("jitnn.cpu.stats");
    /** With {@code jitnn.cpu.stats}: one row per operator and attributes (shapes), not per operator. */
    private static final boolean BY_SHAPE = Boolean.getBoolean("jitnn.cpu.stats.shapes");
    /** Per operator name: calls, nanoseconds (with {@link #STATS}), calls that ran the reference. */
    private static final Map<String, LongAdder[]> BY_OP = new ConcurrentHashMap<>();

    static {
        if (System.getProperty("jitnn.hostPool.bytes") == null) {
            HostPool.raiseCap(Long.getLong("jitnn.cpu.poolBytes", 2L << 30));
        }
        if (STATS) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> System.err.print(report()), "jitnn-cpu-stats"));
        }
    }

    private CpuExecutor() {
    }

    /** Reuse the arrays of a trace's previous execution ({@code -Djitnn.cpu.reuse=false} turns it off). */
    private static final boolean REUSE = !"false".equals(System.getProperty("jitnn.cpu.reuse"));
    /** Trace structures remembered per thread. */
    private static final int REMEMBERED = Integer.getInteger("jitnn.cpu.reuse.traces", 32);

    /**
     * The previous execution of one trace structure: per storage slot, the storage it produced (weakly:
     * a storage nobody holds returns its array to the pool when collected) and whether that sync
     * requested it.
     */
    private static final class Previous {
        final java.lang.ref.WeakReference<Storage>[] storages;
        final boolean[] wanted;

        @SuppressWarnings("unchecked")
        Previous(int slots) {
            storages = new java.lang.ref.WeakReference[slots];
            wanted = new boolean[slots];
        }
    }

    private static final ThreadLocal<java.util.LinkedHashMap<String, Previous>> PREVIOUS = ThreadLocal.withInitial(() -> new java.util.LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, Previous> eldest) {
            return size() > REMEMBERED;
        }
    });

    static void run(List<Node> recorded, Tensor[] wanted) {
        List<Node> nodes = org.beehive.jitnn.ops.Fusion.rewrite(recorded, wanted);
        Set<Storage> wantedSet = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor t : wanted) {
            wantedSet.add(t.storage());
        }
        // Storage slots in order of first appearance, and the trace's structure: a later execution of
        // the same structure (the next training step) takes the arrays of this one's intermediates, as
        // a compiled plan reuses its buffers on a device; those it did not request at the sync go stale.
        IdentityHashMap<Storage, Integer> slotOf = new IdentityHashMap<>();
        List<Storage> slots = new java.util.ArrayList<>();
        StringBuilder key = new StringBuilder(64 * nodes.size());
        for (Node n : nodes) {
            key.append(n.op.name()).append(java.util.Arrays.toString(n.iattr));
            for (Tensor t : n.inputs) {
                key.append('<').append(slot(slotOf, slots, t.storage())).append(t.storage().persistent ? 'P' : ':').append(t.storage().numel);
            }
            for (Tensor t : n.outputs) {
                key.append('>').append(slot(slotOf, slots, t.storage())).append(t.storage().persistent ? 'P' : ':').append(t.storage().numel);
            }
            key.append('|');
        }
        String structure = key.toString();
        Previous previous = REUSE ? PREVIOUS.get().get(structure) : null;
        // the last node that reads or writes each storage, and the storages a node reads
        IdentityHashMap<Storage, Integer> lastUse = new IdentityHashMap<>();
        Set<Storage> produced = Collections.newSetFromMap(new IdentityHashMap<>());
        Set<Storage> consumed = Collections.newSetFromMap(new IdentityHashMap<>());
        for (int i = 0; i < nodes.size(); i++) {
            Node n = nodes.get(i);
            for (Tensor t : n.inputs) {
                lastUse.put(t.storage(), i);
                consumed.add(t.storage());
            }
            for (Tensor t : n.outputs) {
                lastUse.put(t.storage(), i);
                produced.add(t.storage());
            }
        }
        // releases[i]: the storages whose buffers return to the pool after node i
        @SuppressWarnings("unchecked")
        List<Storage>[] releases = new List[nodes.size()];
        for (Map.Entry<Storage, Integer> e : lastUse.entrySet()) {
            Storage s = e.getKey();
            if (!s.persistent && s.internal && produced.contains(s) && consumed.contains(s) && !wantedSet.contains(s)) {
                int at = e.getValue();
                if (releases[at] == null) {
                    releases[at] = new java.util.ArrayList<>();
                }
                releases[at].add(s);
            }
        }
        List<TornadoNativeArray> pooled = new java.util.ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            Node n = nodes.get(i);
            long start = STATS ? System.nanoTime() : 0;
            for (Tensor t : n.inputs) {
                Engine.ensureHost(t.storage());
            }
            pooled.clear();
            for (Tensor t : n.outputs) {
                Storage s = t.storage();
                if (s.array == null && s.heap == null && !s.persistent) {
                    // a pooled or reused array holds old contents: CPU kernels write every element;
                    // the reference gets zeros
                    if (!steal(previous, slotOf, s)) {
                        TornadoNativeArray a = HostPool.take(s.dtype, s.numel);
                        s.array = a;
                        s.lease = HostPool.lease(s, a);
                    }
                    pooled.add(s.array);
                } else {
                    s.allocate();
                }
                s.hostWritten();
            }
            if (STATS) {
                long now = System.nanoTime();
                count("(outputs)", now - start, false);
                start = now;
            }
            boolean kernel = n.op.cpu(n);
            if (!kernel) {
                for (TornadoNativeArray a : pooled) {
                    zero(a.getSegment());
                }
                n.op.reference(n);
            }
            count(BY_SHAPE ? n.op.name() + java.util.Arrays.toString(n.iattr) : n.op.name(), STATS ? System.nanoTime() - start : 0, !kernel);
            n.executed = true;
            if (releases[i] != null) {
                for (Storage s : releases[i]) {
                    recycle(s);
                }
            }
        }
        if (REUSE) {
            Previous now = new Previous(slots.size());
            for (int i = 0; i < slots.size(); i++) {
                Storage s = slots.get(i);
                if (!s.persistent && produced.contains(s)) {
                    now.storages[i] = new java.lang.ref.WeakReference<>(s);
                    now.wanted[i] = wantedSet.contains(s);
                }
            }
            PREVIOUS.get().put(structure, now);
        }
    }

    private static int slot(IdentityHashMap<Storage, Integer> slotOf, List<Storage> slots, Storage s) {
        Integer i = slotOf.get(s);
        if (i == null) {
            i = slots.size();
            slotOf.put(s, i);
            slots.add(s);
        }
        return i;
    }

    /**
     * Gives {@code s} the array its slot had in the previous execution of the same trace, if that
     * storage was not requested at its sync, is not read by this execution, and still holds the
     * array; the previous storage goes stale. True when it did.
     */
    private static boolean steal(Previous previous, IdentityHashMap<Storage, Integer> slotOf, Storage s) {
        if (previous == null) {
            return false;
        }
        int i = slotOf.get(s);
        Storage prev = previous.storages[i] == null ? null : previous.storages[i].get();
        if (prev == null || prev == s || previous.wanted[i] || prev.persistent || prev.array == null || prev.lease == null || prev.dropped || slotOf.containsKey(prev)
                || prev.dtype != s.dtype || prev.numel != s.numel) {
            return false;
        }
        HostPool.Lease lease = prev.lease.transfer(s);
        if (lease == null) {
            return false;
        }
        s.array = prev.array;
        s.lease = lease;
        prev.lease = null;
        prev.array = null;
        prev.hostValid = false;
        prev.stale = true;
        return true;
    }

    private static void count(String name, long nanos, boolean reference) {
        LongAdder[] c = BY_OP.computeIfAbsent(name, k -> new LongAdder[] { new LongAdder(), new LongAdder(), new LongAdder() });
        c[0].increment();
        c[1].add(nanos);
        if (reference) {
            c[2].increment();
        }
    }

    /** Zeroes {@code s} in parallel chunks. */
    private static void zero(java.lang.foreign.MemorySegment s) {
        long bytes = s.byteSize();
        int chunks = (int) Math.min(Integer.MAX_VALUE, (bytes + (1 << 20) - 1) >> 20);
        org.beehive.jitnn.cpu.Cpu.parallelFor(chunks, 1, (from, to) -> {
            long a = (long) from << 20;
            long b = Math.min(bytes, (long) to << 20);
            s.asSlice(a, b - a).fill((byte) 0);
        });
    }

    /** Gives an internal intermediate's buffer back to the pool; reading the storage afterwards is an error. */
    private static void recycle(Storage s) {
        if (s.lease == null) {
            return;
        }
        s.lease.run();
        s.lease = null;
        s.array = null;
        s.hostValid = false;
        s.stale = true;
    }

    /** Per operator name: {@code [calls, nanoseconds (0 without jitnn.cpu.stats), reference fallbacks]} so far. */
    static Map<String, long[]> stats() {
        Map<String, long[]> out = new TreeMap<>();
        BY_OP.forEach((k, v) -> out.put(k, new long[] { v[0].sum(), v[1].sum(), v[2].sum() }));
        return out;
    }

    /** The table printed at exit (operators by time; empty when nothing ran), and the counters start again from zero. */
    static String drainReport() {
        String r = report();
        BY_OP.clear();
        return r;
    }

    private static String report() {
        Map<String, long[]> s = stats();
        if (s.isEmpty()) {
            return "";
        }
        long total = 0;
        for (long[] v : s.values()) {
            total += v[1];
        }
        StringBuilder b = new StringBuilder(String.format(java.util.Locale.ROOT, "[jitnn] CPU backend, %d threads, %d FP32 lanes: operators by time%n",
                org.beehive.jitnn.cpu.Cpu.threads(), org.beehive.jitnn.cpu.Cpu.lanes()));
        long all = Math.max(1, total);
        s.entrySet().stream().sorted((x, y) -> Long.compare(y.getValue()[1], x.getValue()[1])).forEach(e -> {
            long[] v = e.getValue();
            b.append(String.format(java.util.Locale.ROOT, "  %-28s %8d calls %12.3f ms %5.1f%%%s%n", e.getKey(), v[0], v[1] / 1e6, 100.0 * v[1] / all,
                    v[2] > 0 ? "  reference: " + v[2] : ""));
        });
        b.append(String.format(java.util.Locale.ROOT, "  total %.3f ms%n", total / 1e6));
        return b.toString();
    }
}
