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
 */package ai.beestream.jitnn.engine;

import static ai.beestream.jitnn.engine.TornadoExecutor.*;
import static ai.beestream.jitnn.engine.PlanMemo.*;
import static ai.beestream.jitnn.engine.Telemetry.*;
import static ai.beestream.jitnn.engine.PlanCache.*;
import static ai.beestream.jitnn.engine.PlanRecovery.*;
import static ai.beestream.jitnn.engine.PlanRunner.*;

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
 * Compiles a trace into TornadoVM task graphs: kernels, transfers, buffer reuse and the split
 * into a chain of graphs.
 */
final class PlanCompiler {

    private PlanCompiler() {
    }

    static final AtomicInteger GRAPHS = new AtomicInteger();
    static CompiledTrace compile(String key, TraceLayout l, int[] carry) {
        return compile(key, l.nodes(), l.slots(), l.slotOf(), l.kinds(), l.written(), l.recyclable(), l.lastUse(), l.unread(), carry);
    }

    static CompiledTrace compile(String key, List<Node> nodes, List<Storage> slots, IdentityHashMap<Storage, Integer> slotOf, int[] kinds, boolean[] written, boolean[] recyclable,
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
    static String describe(String name, TornadoExecutionPlan plan, List<Node> nodes, int[] tasksPerNode, int graphs) {
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
}
