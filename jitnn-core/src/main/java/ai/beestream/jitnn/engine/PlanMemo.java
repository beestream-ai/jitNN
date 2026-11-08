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
import static ai.beestream.jitnn.engine.Telemetry.*;
import static ai.beestream.jitnn.engine.PlanCompiler.*;
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
 * The memo fast path: a trace recorded again with the same operators, shapes, storage kinds and
 * weight versions replays the plan it ran last time, without being rewritten and keyed again.
 */
final class PlanMemo {

    private PlanMemo() {
    }

    /**
     * Remember traces already run (default on): a trace recorded again with the same operators,
     * shapes, storage kinds and parameter versions replays its plan without being rewritten and
     * keyed again, which is most of jitNN's host time per call at small batches.
     */
    static final boolean MEMO = Boolean.parseBoolean(System.getProperty("jitnn.plan.memo", "true"));
    static final int MEMO_MAX = 64;
    static final Map<Integer, List<Memo>> MEMOS = new java.util.HashMap<>();
    static int memoCount;
    static int memoStamp;
    static long memoReplays;

    /**
     * One thread's signature of the trace it runs: the lock is released while a plan executes, so
     * threads on other devices sign their traces meanwhile.
     */
    static final class Signed {
        long[] sig = new long[4096];
        int len;
        Op[] ops = new Op[256];
        Storage[] locals = new Storage[512];
        boolean[] produced = new boolean[512];
        long[] versions = new long[512];
        int count;
        int stamp;
    }

    static final ThreadLocal<Signed> SIGNED = ThreadLocal.withInitial(Signed::new);

    /**
     * A trace run before: its signature, the persistent storages it read (by local number) at the
     * versions it saw, and its plan's slots as local numbers of the trace's storages, or storages
     * the rewrite made (folded weights, channels-last copies), which are internal and kept.
     */
    record Memo(long[] sig, Op[] ops, Storage[] persistent, long[] versions, boolean[] writtenLocal, boolean[] localInSlots, String global, String key,
            CompiledTrace ct, int[] slotLocal, Storage[] slotFixed, boolean[] written) {
    }

    static void put(Signed sg, long v) {
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
    static boolean signature(Signed sg, List<Node> recorded, Tensor[] wanted) {
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

    static void signTensor(Signed sg, Tensor t, boolean output) {
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

    static int memoHash(Signed sg) {
        int h = sg.count;
        for (int i = 0; i < sg.len; i++) {
            h = 31 * h + Long.hashCode(sg.sig[i]);
        }
        return h;
    }

    /** Runs the remembered plan of the trace just signed, if there is one that still holds. */
    static boolean replay(Signed sg, int hash, String global, List<Node> recorded, Tensor[] wanted) {
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
            // weights another thread's plan is writing are waited for, then the versions checked again
            boolean waited = false;
            for (Storage p : m.persistent()) {
                if (p != null && p.writer != null && p.writer != ct) {
                    awaitWritten(p, ct);
                    waited = true;
                }
            }
            if (waited && (!matches(sg, m, global, recorded.size()) || ct.closed || CACHE.get(m.key()) != ct)) {
                return false;
            }
            replays++;
            memoReplays++;
            ct.lastUsed = ++executionsTotal;
            lastIntermediateBytes = ct.intermediateBytes;
            evictIdle(ct);
            try {
                execute(ct, slots);
            } catch (RuntimeException failure) {
                // out of device memory, or a library failing: the plan is not run again; the full
                // path compiles the trace afresh and handles the failure
                CACHE.remove(m.key());
                closeFailed(ct);
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

    static boolean matches(Signed sg, Memo m, String global, int nodes) {
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
    static void remember(Signed sg, int hash, String global, String key, CompiledTrace ct, List<Storage> slots, boolean[] written, int nodes) {
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
}
