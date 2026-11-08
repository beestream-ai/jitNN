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
package ai.beestream.jitnn.engine;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

import ai.beestream.jitnn.JitnnConfig;
import ai.beestream.jitnn.Tensor;

import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * Runs the operators recorded since the last sync on the CPU backend: the trace is rewritten by the
 * same fusion pass as on a device (the {@link TracePass}es), then each node runs in program order with
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

    static final boolean STATS = JitnnConfig.current().get(JitnnConfig.CPU_STATS);
    /** With {@code jitnn.cpu.stats}: one row per operator and attributes (shapes), not per operator. */
    private static final boolean BY_SHAPE = JitnnConfig.current().get(JitnnConfig.CPU_STATS_SHAPES);
    /** Per operator name: calls, nanoseconds (with {@link #STATS}), calls that ran the reference, calls that ran on the host's OpenCL device. */
    private static final Map<String, LongAdder[]> BY_OP = new ConcurrentHashMap<>();

    static {
        if (!JitnnConfig.current().isSet(JitnnConfig.HOST_POOL_BYTES)) {
            HostPool.raiseCap(JitnnConfig.current().get(JitnnConfig.CPU_POOL_BYTES));
        }
        if (STATS) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> System.err.print(report()), "jitnn-cpu-stats"));
        }
    }

    private CpuExecutor() {
    }

    /** Reuse the arrays of a trace's previous execution ({@code -Djitnn.cpu.reuse=false} turns it off). */
    private static final boolean REUSE = JitnnConfig.current().get(JitnnConfig.CPU_REUSE);
    /** Trace structures remembered per thread. */
    private static final int REMEMBERED = JitnnConfig.current().get(JitnnConfig.CPU_REUSE_TRACES);

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
        List<Node> nodes = TracePass.applyAll(recorded, wanted);
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
        for (int i = 0; i < nodes.size();) {
            int end = i + 1;
            Choice choice = choice(nodes.get(i));
            if (choice != null && choice.onDevice()) {
                // operators without a CPU kernel, run together on the host's OpenCL device (one at a
                // time while they are being timed)
                while (!choice.measuring() && end < nodes.size()) {
                    Choice next = choice(nodes.get(end));
                    if (next == null || !next.onDevice() || next.measuring()) {
                        break;
                    }
                    end++;
                }
                long nanos = openCl(nodes.subList(i, end), nodes, end, wantedSet);
                if (nanos >= 0) {
                    choice.deviceRan(nanos);
                } else {
                    for (int j = i; j < end; j++) {
                        Node n = nodes.get(j);
                        if (end - i > 1 && openCl(List.of(n), nodes, j + 1, wantedSet) >= 0) {
                            continue;
                        }
                        // the device cannot compile it (no FP16 on the device, say): the reference runs it from now on
                        choice(n).failed();
                        detach(n);
                        host(n, previous, slotOf, pooled);
                    }
                }
            } else {
                long start = choice != null ? System.nanoTime() : 0;
                host(nodes.get(i), previous, slotOf, pooled);
                if (choice != null) {
                    choice.referenceRan(System.nanoTime() - start);
                }
            }
            for (int j = i; j < end; j++) {
                if (releases[j] != null) {
                    for (Storage s : releases[j]) {
                        recycle(s);
                    }
                }
            }
            i = end;
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

    /** Runs {@code n} on the host: its CPU kernel, else its reference. */
    private static void host(Node n, Previous previous, IdentityHashMap<Storage, Integer> slotOf, List<TornadoNativeArray> pooled) {
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
        count(key(n), STATS ? System.nanoTime() - start : 0, !kernel);
        n.executed = true;
    }

    private static String key(Node n) {
        return BY_SHAPE ? n.op.name() + java.util.Arrays.toString(n.iattr) : n.op.name();
    }

    /** Operator classes that override {@link Op#cpu}. */
    private static final ClassValue<Boolean> HAS_KERNEL = new ClassValue<>() {
        @Override
        protected Boolean computeValue(Class<?> type) {
            try {
                return type.getMethod("cpu", Node.class).getDeclaringClass() != Op.class;
            } catch (NoSuchMethodException e) {
                return false;
            }
        }
    };

    /** Per operator and attributes: where an operator without a CPU kernel runs. */
    private static final Map<String, Choice> CHOICES = new ConcurrentHashMap<>();

    /** Where {@code n} runs when its operator has no CPU kernel and the host has an OpenCL device; null otherwise. */
    private static Choice choice(Node n) {
        if (HAS_KERNEL.get(n.op.getClass()) || Devices.hostOpenCl() == null) {
            return null;
        }
        return CHOICES.computeIfAbsent(n.op.name() + java.util.Arrays.toString(n.iattr), Choice::new);
    }

    /**
     * Where an operator without a CPU kernel runs: its reference runs twice (the first warms the JIT)
     * and the host's OpenCL device three times (the first compiles), timed, and the faster place keeps
     * it; inputs copied in and outputs copied back count against the device. {@code -Djitnn.cpu.opencl.choose=always}
     * sends it to the device without timing. An operator the device fails to compile stays with its
     * reference.
     */
    private static final class Choice {
        private static final int MEASURING = 0;
        private static final int DEVICE = 1;
        private static final int REFERENCE = 2;

        private final String key;
        private long reference = Long.MAX_VALUE;
        private long device = Long.MAX_VALUE;
        private int referenceRuns;
        private int deviceRuns;
        private volatile int decided = MEASURING;

        Choice(String key) {
            this.key = key;
        }

        private static boolean always() {
            return "always".equals(JitnnConfig.current().get(JitnnConfig.CPU_OPENCL_CHOOSE));
        }

        synchronized boolean onDevice() {
            if (always()) {
                return decided != REFERENCE;
            }
            return decided == DEVICE || (decided == MEASURING && referenceRuns >= 2);
        }

        boolean measuring() {
            return decided == MEASURING && !always();
        }

        synchronized void referenceRan(long nanos) {
            if (decided == MEASURING && ++referenceRuns == 2) {
                reference = nanos;
            }
        }

        synchronized void deviceRan(long nanos) {
            if (decided != MEASURING || always()) {
                return;
            }
            if (++deviceRuns > 1) {
                device = Math.min(device, nanos);
            }
            if (deviceRuns == 3) {
                decided = device < reference ? DEVICE : REFERENCE;
                if (JitnnConfig.current().get(JitnnConfig.VERBOSE)) {
                    System.err.printf("[jitnn] %s runs %s: %.3f ms on the host's OpenCL device, %.3f ms as reference%n", key, decided == DEVICE ? "on the OpenCL device" : "its reference",
                            device / 1e6, reference / 1e6);
                }
            }
        }

        void failed() {
            decided = REFERENCE;
        }
    }

    /**
     * Runs {@code segment} (the nodes before {@code nodes[end]}) as one TornadoVM plan on the host's
     * OpenCL device; the outputs a later node reads, the sync requested or user code holds come back
     * to host arrays. Returns the nanoseconds it took, or -1, with nothing written, when the device
     * failed to compile it.
     */
    private static long openCl(List<Node> segment, List<Node> nodes, int end, Set<Storage> wantedSet) {
        long start = System.nanoTime();
        Set<Storage> later = Collections.newSetFromMap(new IdentityHashMap<>());
        for (int j = end; j < nodes.size(); j++) {
            for (Tensor t : nodes.get(j).inputs) {
                later.add(t.storage());
            }
        }
        List<Tensor> back = new java.util.ArrayList<>();
        for (Node n : segment) {
            for (Tensor t : n.outputs) {
                Storage s = t.storage();
                if (later.contains(s) || wantedSet.contains(s) || !s.internal) {
                    back.add(t);
                }
            }
        }
        String saved = Devices.current();
        Devices.set(Devices.hostOpenCl());
        try {
            TornadoExecutor.run(segment, back.toArray(new Tensor[0]));
        } catch (RuntimeException e) {
            if (JitnnConfig.current().get(JitnnConfig.VERBOSE)) {
                System.err.printf("[jitnn] the host's OpenCL device did not run %s: %s%n", segment.stream().map(n -> n.op.name()).toList(), e.getMessage());
            }
            return -1;
        } finally {
            Devices.set(saved);
        }
        long nanos = System.nanoTime() - start;
        for (Node n : segment) {
            n.executed = true;
            countOpenCl(key(n), STATS ? nanos / segment.size() : 0);
        }
        return nanos;
    }

    /** Detaches the outputs of a node whose OpenCL plan failed from that plan's arrays. */
    private static void detach(Node n) {
        for (Tensor t : n.outputs) {
            Storage s = t.storage();
            if (!s.persistent && s.lease == null) {
                s.array = null;
                s.boundTrace = null;
            }
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
        LongAdder[] c = counters(name);
        c[0].increment();
        c[1].add(nanos);
        if (reference) {
            c[2].increment();
        }
    }

    private static void countOpenCl(String name, long nanos) {
        LongAdder[] c = counters(name);
        c[0].increment();
        c[1].add(nanos);
        c[3].increment();
    }

    private static LongAdder[] counters(String name) {
        return BY_OP.computeIfAbsent(name, k -> new LongAdder[] { new LongAdder(), new LongAdder(), new LongAdder(), new LongAdder() });
    }

    /** Zeroes {@code s} in parallel chunks. */
    private static void zero(java.lang.foreign.MemorySegment s) {
        long bytes = s.byteSize();
        int chunks = (int) Math.min(Integer.MAX_VALUE, (bytes + (1 << 20) - 1) >> 20);
        ai.beestream.jitnn.cpu.Cpu.parallelFor(chunks, 1, (from, to) -> {
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

    /**
     * Per operator name: {@code [calls, nanoseconds (0 without jitnn.cpu.stats), reference fallbacks,
     * calls on the host's OpenCL device]} so far.
     */
    static Map<String, long[]> stats() {
        Map<String, long[]> out = new TreeMap<>();
        BY_OP.forEach((k, v) -> out.put(k, new long[] { v[0].sum(), v[1].sum(), v[2].sum(), v[3].sum() }));
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
                ai.beestream.jitnn.cpu.Cpu.threads(), ai.beestream.jitnn.cpu.Cpu.lanes()));
        long all = Math.max(1, total);
        s.entrySet().stream().sorted((x, y) -> Long.compare(y.getValue()[1], x.getValue()[1])).forEach(e -> {
            long[] v = e.getValue();
            b.append(String.format(java.util.Locale.ROOT, "  %-28s %8d calls %12.3f ms %5.1f%%%s%n", e.getKey(), v[0], v[1] / 1e6, 100.0 * v[1] / all,
                    (v[2] > 0 ? "  reference: " + v[2] : "") + (v[3] > 0 ? "  OpenCL: " + v[3] : "")));
        });
        b.append(String.format(java.util.Locale.ROOT, "  total %.3f ms%n", total / 1e6));
        long[] gemm = ai.beestream.jitnn.cpu.Gemm.drainTimes();
        if (gemm[0] + gemm[1] > 0) {
            b.append(String.format(java.util.Locale.ROOT, "  of which GEMM operand packing %.3f ms, multiplication %.3f ms (outside per-item parallel loops)%n", gemm[0] / 1e6, gemm[1] / 1e6));
        }
        return b.toString();
    }
}
