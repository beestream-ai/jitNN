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

import static ai.beestream.jitnn.engine.TornadoExecutor.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import ai.beestream.jitnn.Tensor;

/**
 * A rewritten trace laid out for compiling: its storages in order of first appearance (the plan's
 * slots), each slot's kind (persistent, input, intermediate), which persistent storages it writes,
 * which intermediates may hand their buffer on (buffer reuse), and the cache key: the structure
 * (operators, attributes, shapes, kinds) and the persistent storages' identities.
 *
 * @param structure the cache key without the persistent storages' identities (plans of the same
 *                  structure can be reused for other weights)
 * @param ids       the persistent storages' identities, appended to the structure for the full key
 */
record TraceLayout(List<Node> nodes, List<Storage> slots, IdentityHashMap<Storage, Integer> slotOf, int[] kinds, boolean[] written, boolean[] recyclable, int[] lastUse,
        Set<Storage> unread, String structure, String ids) {

    /** Lays out {@code nodes} (the rewritten trace); {@code prefix} starts the key (device, slot, libraries, tuning). */
    static TraceLayout of(List<Node> nodes, Tensor[] wanted, String prefix) {
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
        StringBuilder key = new StringBuilder(256).append(prefix);
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
        return new TraceLayout(nodes, slots, slotOf, kinds, written, recyclable, lastUse, unread, key.toString(), ids.toString());
    }
}
