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
package ai.beestream.jitnn.cpu;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * oneDNN's convolutions (forward with bias and ReLU, backward data, backward weights), called
 * through the foreign function API: {@code -Djitnn.cpu.onednn=<library file>} names the library
 * ({@code libdnnl.so.3}), {@code auto} (the default) looks for it on the system's library path,
 * {@code off} uses none. Use a build with OpenMP threads: jitNN sets {@link Cpu#threads()} OpenMP
 * threads for each thread that calls it (a TBB build takes every core and cannot be told otherwise).
 *
 * <p>Each convolution shape gets a plan: oneDNN picks the memory layouts its fastest
 * implementation wants, and the plan reorders jitNN's NCHW tensors into them and the output back;
 * a weight whose version did not change since the last call is not reordered again.
 */
public final class Dnnl {

    private static final ValueLayout.OfInt I = ValueLayout.JAVA_INT;
    private static final ValueLayout.OfLong J = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfFloat F = ValueLayout.JAVA_FLOAT;
    private static final ValueLayout A = ValueLayout.ADDRESS;
    private static final MemoryLayout EXEC_ARG = MemoryLayout.structLayout(I, MemoryLayout.paddingLayout(4), A);

    // oneDNN's constants (dnnl_types.h, dnnl_common_types.h)
    private static final int CPU = 1;
    private static final int IN_ORDER = 1;
    private static final int F32 = 3;
    private static final int TAG_ANY = 1;
    private static final int TAG_A = 2;
    private static final int TAG_ABCD = 5;
    private static final int TAG_ABCDE = 6;
    private static final int FORWARD_TRAINING = 64;
    private static final int FORWARD_INFERENCE = 96;
    private static final int CONVOLUTION_DIRECT = 1;
    private static final int ELTWISE_RELU = 0x20;
    private static final int QUERY_SRC = 129;
    private static final int QUERY_DIFF_SRC = 130;
    private static final int QUERY_WEIGHTS = 131;
    private static final int QUERY_DIFF_WEIGHTS = 132;
    private static final int QUERY_DST = 133;
    private static final int QUERY_DIFF_DST = 134;
    private static final int ARG_SRC = 1;
    private static final int ARG_DST = 17;
    private static final int ARG_WEIGHTS = 33;
    private static final int ARG_BIAS = 41;
    private static final int ARG_DIFF_SRC = 129;
    private static final int ARG_DIFF_DST = 145;
    private static final int ARG_DIFF_WEIGHTS = 161;
    private static final MemorySegment ALLOCATE = MemorySegment.ofAddress(-1L);

    private static final String[] AUTO = { "libdnnl.so.3", "libdnnl.so", "libdnnl.3.dylib", "libdnnl.dylib" };

    private static final String NAME;
    private static MethodHandle engineCreate, streamCreate, streamWait, mdCreate, mdEqual, pdQueryMd, convForward, convBackwardData, convBackwardWeights, reorderPd, primitiveCreate,
            execute, memoryCreate, memorySetHandle, attrCreate, attrSetPostOps, postOpsCreate, postOpsAppendEltwise, ompSetThreads;
    private static MemorySegment engine;

    static {
        String name = null;
        String p = System.getProperty("jitnn.cpu.onednn", "auto").trim();
        if (!p.equals("off")) {
            for (String candidate : p.equals("auto") ? AUTO : new String[] { p }) {
                try {
                    bind(candidate);
                    name = candidate;
                    break;
                } catch (Throwable e) {
                    if (!p.equals("auto")) {
                        System.err.println("[jitnn] cannot use oneDNN library " + candidate + ": " + e);
                    }
                }
            }
        }
        NAME = name;
        if (NAME != null && Boolean.getBoolean("jitnn.verbose")) {
            System.err.println("[jitnn] the CPU backend calls the convolutions of " + NAME + (ompSetThreads != null ? " (" + Cpu.threads() + " OpenMP threads)" : ""));
        }
    }

    private Dnnl() {
    }

    private static void bind(String library) throws Throwable {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lib = library.contains("/") && Files.exists(Path.of(library)) ? SymbolLookup.libraryLookup(Path.of(library), Arena.global())
                : SymbolLookup.libraryLookup(library, Arena.global());
        engineCreate = handle(linker, lib, "dnnl_engine_create", FunctionDescriptor.of(I, A, I, J));
        streamCreate = handle(linker, lib, "dnnl_stream_create", FunctionDescriptor.of(I, A, A, I));
        streamWait = handle(linker, lib, "dnnl_stream_wait", FunctionDescriptor.of(I, A));
        mdCreate = handle(linker, lib, "dnnl_memory_desc_create_with_tag", FunctionDescriptor.of(I, A, I, A, I, I));
        mdEqual = handle(linker, lib, "dnnl_memory_desc_equal", FunctionDescriptor.of(I, A, A));
        pdQueryMd = handle(linker, lib, "dnnl_primitive_desc_query_md", FunctionDescriptor.of(A, A, I, I));
        convForward = handle(linker, lib, "dnnl_convolution_forward_primitive_desc_create", FunctionDescriptor.of(I, A, A, I, I, A, A, A, A, A, A, A, A, A));
        convBackwardData = handle(linker, lib, "dnnl_convolution_backward_data_primitive_desc_create", FunctionDescriptor.of(I, A, A, I, A, A, A, A, A, A, A, A, A));
        convBackwardWeights = handle(linker, lib, "dnnl_convolution_backward_weights_primitive_desc_create", FunctionDescriptor.of(I, A, A, I, A, A, A, A, A, A, A, A, A, A));
        reorderPd = handle(linker, lib, "dnnl_reorder_primitive_desc_create", FunctionDescriptor.of(I, A, A, A, A, A, A));
        primitiveCreate = handle(linker, lib, "dnnl_primitive_create", FunctionDescriptor.of(I, A, A));
        execute = handle(linker, lib, "dnnl_primitive_execute", FunctionDescriptor.of(I, A, A, I, A));
        memoryCreate = handle(linker, lib, "dnnl_memory_create", FunctionDescriptor.of(I, A, A, A, A));
        memorySetHandle = handle(linker, lib, "dnnl_memory_set_data_handle", FunctionDescriptor.of(I, A, A));
        attrCreate = handle(linker, lib, "dnnl_primitive_attr_create", FunctionDescriptor.of(I, A));
        attrSetPostOps = handle(linker, lib, "dnnl_primitive_attr_set_post_ops", FunctionDescriptor.of(I, A, A));
        postOpsCreate = handle(linker, lib, "dnnl_post_ops_create", FunctionDescriptor.of(I, A));
        postOpsAppendEltwise = handle(linker, lib, "dnnl_post_ops_append_eltwise", FunctionDescriptor.of(I, A, I, F, F));
        for (String omp : new String[] { "libgomp.so.1", "libiomp5.so", "libomp.so", "libomp.dylib", "libiomp5.dylib" }) {
            try {
                SymbolLookup o = SymbolLookup.libraryLookup(omp, Arena.global());
                if (o.find("omp_set_num_threads").isPresent()) {
                    ompSetThreads = handle(linker, o, "omp_set_num_threads", FunctionDescriptor.ofVoid(I));
                    break;
                }
            } catch (IllegalArgumentException e) {
                // not this OpenMP runtime
            }
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(A);
            check((int) engineCreate.invokeExact(out, CPU, 0L), "engine");
            engine = out.get(ValueLayout.ADDRESS, 0);
        }
    }

    private static MethodHandle handle(Linker linker, SymbolLookup lib, String name, FunctionDescriptor fd) {
        return linker.downcallHandle(lib.find(name).orElseThrow(() -> new IllegalArgumentException("no " + name)), fd);
    }

    private static void check(int status, String what) {
        if (status != 0) {
            throw new IllegalStateException("oneDNN " + what + " failed with status " + status);
        }
    }

    /** True when oneDNN was found. */
    public static boolean available() {
        return NAME != null;
    }

    /** The library's file, or null. */
    public static String library() {
        return NAME;
    }

    private static final ThreadLocal<MemorySegment> STREAM = ThreadLocal.withInitial(() -> {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(A);
            check((int) streamCreate.invokeExact(out, engine, IN_ORDER), "stream");
            if (ompSetThreads != null) {
                // OpenMP's thread count is per calling thread
                ompSetThreads.invokeExact(Cpu.threads());
            }
            return out.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable e) {
            throw new IllegalStateException(e);
        }
    });

    private static final Map<String, Plan> PLANS = new ConcurrentHashMap<>();

    /**
     * y = act(conv2d(x, w) + bias) in NCHW, grouped when {@code groups > 1}; {@code owner} and
     * {@code version} identify the weight (null owner: reorder it on every call).
     */
    public static void conv2d(MemorySegment x, MemorySegment w, Object owner, long version, MemorySegment bias, MemorySegment y, int n, int cin, int h, int wd, int cout, int oh, int ow,
            int r, int stride, int pad, int dil, int groups, boolean relu) {
        String key = "fwd " + shape(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups) + (bias != null ? " bias" : "") + (relu ? " relu" : "");
        Plan plan = PLANS.computeIfAbsent(key, k -> Plan.forward(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, bias != null, relu));
        plan.run(new MemorySegment[] { x, w, bias, y }, owner, version);
    }

    /** dx = the gradient of conv2d with respect to its input, from dy and the weight (NCHW). */
    public static void conv2dBackwardData(MemorySegment dy, MemorySegment w, Object owner, long version, MemorySegment dx, int n, int cin, int h, int wd, int cout, int oh, int ow, int r,
            int stride, int pad, int dil, int groups) {
        String key = "bwd_data " + shape(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups);
        Plan plan = PLANS.computeIfAbsent(key, k -> Plan.backwardData(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups));
        plan.run(new MemorySegment[] { dy, w, dx }, owner, version);
    }

    /** dw = the gradient of conv2d with respect to its weight, from dy and the input (NCHW). */
    public static void conv2dBackwardWeights(MemorySegment dy, MemorySegment x, MemorySegment dw, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil,
            int groups) {
        String key = "bwd_weights " + shape(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups);
        Plan plan = PLANS.computeIfAbsent(key, k -> Plan.backwardWeights(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups));
        plan.run(new MemorySegment[] { x, dy, dw }, null, 0);
    }

    private static String shape(int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
        return n + "x" + cin + "x" + h + "x" + wd + " " + cout + "x" + oh + "x" + ow + " r" + r + " s" + stride + " p" + pad + " d" + dil + " g" + groups;
    }

    /**
     * One tensor of a primitive: a memory object over jitNN's array (the plain layout), and, when the
     * primitive wants another layout, one over the library's own buffer and the reorder between them.
     */
    private static final class Port {
        final int arg;
        final boolean input;
        final MemorySegment user;
        final MemorySegment inner;
        final MemorySegment reorder;
        final MemorySegment reorderArgs;

        Port(Arena arena, int arg, boolean input, MemorySegment userMd, MemorySegment primitiveMd) throws Throwable {
            this.arg = arg;
            this.input = input;
            user = memory(arena, userMd, MemorySegment.NULL);
            if ((int) mdEqual.invokeExact(userMd, primitiveMd) != 0) {
                inner = user;
                reorder = null;
                reorderArgs = null;
            } else {
                inner = memory(arena, primitiveMd, ALLOCATE);
                reorder = primitive(arena, input ? reorderPd(arena, userMd, primitiveMd) : reorderPd(arena, primitiveMd, userMd));
                reorderArgs = args(arena, new int[] { ARG_SRC, ARG_DST }, input ? new MemorySegment[] { user, inner } : new MemorySegment[] { inner, user });
            }
        }

        void reorder(MemorySegment stream) throws Throwable {
            if (reorder != null) {
                check((int) execute.invokeExact(reorder, stream, 2, reorderArgs), "reorder");
            }
        }
    }

    private static final class Plan {
        final Arena arena = Arena.ofShared();
        final List<Port> ports = new ArrayList<>();
        /** The index in {@link #ports} of the weight reordered only when it changes, or -1. */
        int weight = -1;
        MemorySegment primitive;
        MemorySegment args;
        Object lastOwner;
        long lastVersion;

        static Plan forward(int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups, boolean bias, boolean relu) {
            Plan p = new Plan();
            try {
                Arena a = p.arena;
                MemorySegment src = md(a, TAG_ABCD, n, cin, h, wd);
                MemorySegment wts = weightsMd(a, cout, cin, r, groups);
                MemorySegment dst = md(a, TAG_ABCD, n, cout, oh, ow);
                MemorySegment b = bias ? md(a, TAG_A, cout) : MemorySegment.NULL;
                MemorySegment attr = MemorySegment.NULL;
                if (relu) {
                    MemorySegment out = a.allocate(A);
                    check((int) postOpsCreate.invokeExact(out), "post-ops");
                    MemorySegment ops = out.get(ValueLayout.ADDRESS, 0);
                    check((int) postOpsAppendEltwise.invokeExact(ops, ELTWISE_RELU, 0f, 0f), "post-op ReLU");
                    check((int) attrCreate.invokeExact(out), "attributes");
                    attr = out.get(ValueLayout.ADDRESS, 0);
                    check((int) attrSetPostOps.invokeExact(attr, ops), "post-ops");
                }
                MemorySegment pd = convForwardPd(a, FORWARD_INFERENCE, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, b, attr);
                p.add(new Port(a, ARG_SRC, true, src, query(pd, QUERY_SRC)));
                p.weight = p.ports.size();
                p.add(new Port(a, ARG_WEIGHTS, true, wts, query(pd, QUERY_WEIGHTS)));
                if (bias) {
                    p.add(new Port(a, ARG_BIAS, true, b, b));
                } else {
                    p.ports.add(null);
                }
                p.add(new Port(a, ARG_DST, false, dst, query(pd, QUERY_DST)));
                p.finish(pd);
                return p;
            } catch (Throwable e) {
                throw failed(e);
            }
        }

        static Plan backwardData(int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
            Plan p = new Plan();
            try {
                Arena a = p.arena;
                MemorySegment hint = convForwardPd(a, FORWARD_TRAINING, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, MemorySegment.NULL, MemorySegment.NULL);
                MemorySegment out = a.allocate(A);
                long[] dims = dims(stride, pad, dil, h, wd, oh, ow, r);
                check((int) convBackwardData.invokeExact(out, engine, CONVOLUTION_DIRECT, md(a, TAG_ANY, n, cin, h, wd), weightsMd(a, cout, cin, r, groups, TAG_ANY),
                        md(a, TAG_ANY, n, cout, oh, ow), vec(a, dims[0], dims[1]), vec(a, dims[2], dims[3]), vec(a, dims[4], dims[5]), vec(a, dims[6], dims[7]), hint,
                        MemorySegment.NULL), "backward data descriptor");
                MemorySegment pd = out.get(ValueLayout.ADDRESS, 0);
                p.add(new Port(a, ARG_DIFF_DST, true, md(a, TAG_ABCD, n, cout, oh, ow), query(pd, QUERY_DIFF_DST)));
                p.weight = p.ports.size();
                p.add(new Port(a, ARG_WEIGHTS, true, weightsMd(a, cout, cin, r, groups), query(pd, QUERY_WEIGHTS)));
                p.add(new Port(a, ARG_DIFF_SRC, false, md(a, TAG_ABCD, n, cin, h, wd), query(pd, QUERY_DIFF_SRC)));
                p.finish(pd);
                return p;
            } catch (Throwable e) {
                throw failed(e);
            }
        }

        static Plan backwardWeights(int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
            Plan p = new Plan();
            try {
                Arena a = p.arena;
                MemorySegment hint = convForwardPd(a, FORWARD_TRAINING, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, MemorySegment.NULL, MemorySegment.NULL);
                MemorySegment out = a.allocate(A);
                long[] dims = dims(stride, pad, dil, h, wd, oh, ow, r);
                check((int) convBackwardWeights.invokeExact(out, engine, CONVOLUTION_DIRECT, md(a, TAG_ANY, n, cin, h, wd), weightsMd(a, cout, cin, r, groups, TAG_ANY),
                        MemorySegment.NULL, md(a, TAG_ANY, n, cout, oh, ow), vec(a, dims[0], dims[1]), vec(a, dims[2], dims[3]), vec(a, dims[4], dims[5]), vec(a, dims[6], dims[7]),
                        hint, MemorySegment.NULL), "backward weights descriptor");
                MemorySegment pd = out.get(ValueLayout.ADDRESS, 0);
                p.add(new Port(a, ARG_SRC, true, md(a, TAG_ABCD, n, cin, h, wd), query(pd, QUERY_SRC)));
                p.add(new Port(a, ARG_DIFF_DST, true, md(a, TAG_ABCD, n, cout, oh, ow), query(pd, QUERY_DIFF_DST)));
                p.add(new Port(a, ARG_DIFF_WEIGHTS, false, weightsMd(a, cout, cin, r, groups), query(pd, QUERY_DIFF_WEIGHTS)));
                p.finish(pd);
                return p;
            } catch (Throwable e) {
                throw failed(e);
            }
        }

        private void add(Port port) {
            ports.add(port);
        }

        private void finish(MemorySegment pd) throws Throwable {
            primitive = primitive(arena, pd);
            List<Port> used = ports.stream().filter(java.util.Objects::nonNull).toList();
            int[] ids = new int[used.size()];
            MemorySegment[] mems = new MemorySegment[used.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = used.get(i).arg;
                mems[i] = used.get(i).inner;
            }
            args = args(arena, ids, mems);
        }

        /** Runs the plan on {@code data} (one array per port, in order). */
        synchronized void run(MemorySegment[] data, Object owner, long version) {
            try {
                MemorySegment stream = STREAM.get();
                for (int i = 0; i < ports.size(); i++) {
                    Port port = ports.get(i);
                    if (port != null) {
                        check((int) memorySetHandle.invokeExact(port.user, data[i]), "data handle");
                    }
                }
                for (int i = 0; i < ports.size(); i++) {
                    Port port = ports.get(i);
                    if (port == null || !port.input) {
                        continue;
                    }
                    if (i == weight && owner != null && owner == lastOwner && version == lastVersion) {
                        // the weight is in the library's layout already
                        continue;
                    }
                    port.reorder(stream);
                }
                lastOwner = owner;
                lastVersion = version;
                check((int) execute.invokeExact(primitive, stream, (int) (args.byteSize() / EXEC_ARG.byteSize()), args), "convolution");
                for (Port port : ports) {
                    if (port != null && !port.input) {
                        port.reorder(stream);
                    }
                }
                check((int) streamWait.invokeExact(stream), "stream wait");
            } catch (RuntimeException e) {
                lastOwner = null;
                throw e;
            } catch (Throwable e) {
                lastOwner = null;
                throw new IllegalStateException(e);
            }
        }
    }

    private static RuntimeException failed(Throwable e) {
        return e instanceof RuntimeException re ? re : new IllegalStateException(e);
    }

    private static MemorySegment convForwardPd(Arena a, int prop, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups,
            MemorySegment bias, MemorySegment attr) throws Throwable {
        MemorySegment out = a.allocate(A);
        long[] dims = dims(stride, pad, dil, h, wd, oh, ow, r);
        check((int) convForward.invokeExact(out, engine, prop, CONVOLUTION_DIRECT, md(a, TAG_ANY, n, cin, h, wd), weightsMd(a, cout, cin, r, groups, TAG_ANY), bias,
                md(a, TAG_ANY, n, cout, oh, ow), vec(a, dims[0], dims[1]), vec(a, dims[2], dims[3]), vec(a, dims[4], dims[5]), vec(a, dims[6], dims[7]), attr), "forward descriptor");
        return out.get(ValueLayout.ADDRESS, 0);
    }

    /** Strides, dilations (oneDNN counts the gaps: 0 is none), left and right padding, for height and width. */
    private static long[] dims(int stride, int pad, int dil, int h, int wd, int oh, int ow, int r) {
        long padBottom = (long) (oh - 1) * stride + (long) (r - 1) * dil + 1 - h - pad;
        long padRight = (long) (ow - 1) * stride + (long) (r - 1) * dil + 1 - wd - pad;
        return new long[] { stride, stride, dil - 1, dil - 1, pad, pad, padBottom, padRight };
    }

    private static MemorySegment weightsMd(Arena a, int cout, int cin, int r, int groups) throws Throwable {
        return groups == 1 ? md(a, TAG_ABCD, cout, cin, r, r) : md(a, TAG_ABCDE, groups, cout / groups, cin / groups, r, r);
    }

    private static MemorySegment weightsMd(Arena a, int cout, int cin, int r, int groups, int tag) throws Throwable {
        return groups == 1 ? md(a, tag, cout, cin, r, r) : md(a, tag, groups, cout / groups, cin / groups, r, r);
    }

    private static MemorySegment md(Arena a, int tag, long... dims) throws Throwable {
        MemorySegment out = a.allocate(A);
        check((int) mdCreate.invokeExact(out, dims.length, vec(a, dims), F32, tag), "memory descriptor");
        return out.get(ValueLayout.ADDRESS, 0);
    }

    private static MemorySegment vec(Arena a, long... values) {
        MemorySegment v = a.allocate(J, 12);
        for (int i = 0; i < values.length; i++) {
            v.setAtIndex(J, i, values[i]);
        }
        return v;
    }

    private static MemorySegment query(MemorySegment pd, int what) throws Throwable {
        MemorySegment md = (MemorySegment) pdQueryMd.invokeExact(pd, what, 0);
        if (md.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("oneDNN has no memory descriptor " + what);
        }
        return md;
    }

    private static MemorySegment memory(Arena a, MemorySegment md, MemorySegment handle) throws Throwable {
        MemorySegment out = a.allocate(A);
        check((int) memoryCreate.invokeExact(out, md, engine, handle), "memory");
        return out.get(ValueLayout.ADDRESS, 0);
    }

    private static MemorySegment reorderPd(Arena a, MemorySegment from, MemorySegment to) throws Throwable {
        MemorySegment out = a.allocate(A);
        check((int) reorderPd.invokeExact(out, from, engine, to, engine, MemorySegment.NULL), "reorder descriptor");
        return out.get(ValueLayout.ADDRESS, 0);
    }

    private static MemorySegment primitive(Arena a, MemorySegment pd) throws Throwable {
        MemorySegment out = a.allocate(A);
        check((int) primitiveCreate.invokeExact(out, pd), "primitive");
        return out.get(ValueLayout.ADDRESS, 0);
    }

    private static MemorySegment args(Arena a, int[] ids, MemorySegment[] mems) {
        MemorySegment s = a.allocate(EXEC_ARG, ids.length);
        for (int i = 0; i < ids.length; i++) {
            s.set(I, i * EXEC_ARG.byteSize(), ids[i]);
            s.set(ValueLayout.ADDRESS, i * EXEC_ARG.byteSize() + 8, mems[i]);
        }
        return s;
    }
}
