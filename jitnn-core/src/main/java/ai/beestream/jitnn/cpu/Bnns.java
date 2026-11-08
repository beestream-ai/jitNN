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

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The convolutions of Apple's BNNS (Accelerate, macOS): forward with bias and ReLU, backward data
 * and backward weights, grouped too, called through the foreign function API on {@link
 * Cpu#threads()} threads. {@code -Djitnn.cpu.bnns=auto} (the default) uses it on macOS, {@code off}
 * never.
 *
 * <p>A forward BNNS filter packs the weights (and the bias) when it is created, so each convolution
 * shape keeps a filter for the weight version it was made from and makes a new one when the weight
 * changes. The backward pass needs filters that read the weights in place
 * ({@code BNNSFlagsUseClientPtr}). The filter API is deprecated since macOS 15 in favour of BNNS graphs, which need a
 * compiled Core ML model; it still runs.
 */
public final class Bnns {

    private static final ValueLayout.OfLong J = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfInt I = ValueLayout.JAVA_INT;
    private static final AddressLayout A = ValueLayout.ADDRESS;

    // bnns_constants.h, bnns_structures.h (offsets checked against the macOS 26 SDK)
    private static final int FLOAT32 = 65568;
    private static final int LAYOUT_CHW = 196608;
    private static final int LAYOUT_OIHW = 262144;
    private static final int LAYOUT_VECTOR = 65536;
    private static final int RELU = 1;
    private static final long ND_SIZE = 176;
    private static final long ND_LAYOUT = 4;
    private static final long ND_SIZES = 8;
    private static final long ND_DATA = 136;
    private static final long ND_TYPE = 144;
    private static final long CONV_SIZE = 840;
    private static final long CONV_W = 176;
    private static final long CONV_O = 352;
    private static final long CONV_BIAS = 528;
    private static final long CONV_ACTIVATION = 704;
    private static final long CONV_X_STRIDE = 752;
    private static final long FILTER_PARAMS_SIZE = 32;

    private static final MethodHandle CREATE;
    private static final MethodHandle APPLY_BATCH;
    private static final MethodHandle APPLY_BACKWARD_BATCH;
    private static final MethodHandle DESTROY;

    static {
        MethodHandle create = null, apply = null, backward = null, destroy = null;
        String p = System.getProperty("jitnn.cpu.bnns", "auto").trim();
        if (!p.equals("off") && System.getProperty("os.name", "").startsWith("Mac")) {
            try {
                Linker linker = Linker.nativeLinker();
                SymbolLookup lib = SymbolLookup.libraryLookup("/System/Library/Frameworks/Accelerate.framework/Accelerate", Arena.global());
                create = linker.downcallHandle(lib.find("BNNSFilterCreateLayerConvolution").orElseThrow(), FunctionDescriptor.of(A, A, A));
                apply = linker.downcallHandle(lib.find("BNNSFilterApplyBatch").orElseThrow(), FunctionDescriptor.of(I, A, J, A, J, A, J));
                backward = linker.downcallHandle(lib.find("BNNSFilterApplyBackwardBatch").orElseThrow(), FunctionDescriptor.of(I, A, J, A, J, A, J, A, J, A, J, A, A));
                destroy = linker.downcallHandle(lib.find("BNNSFilterDestroy").orElseThrow(), FunctionDescriptor.ofVoid(A));
            } catch (Throwable e) {
                create = null;
                if (!p.equals("auto")) {
                    System.err.println("[jitnn] cannot use BNNS: " + e);
                }
            }
        }
        CREATE = create;
        APPLY_BATCH = apply;
        APPLY_BACKWARD_BATCH = backward;
        DESTROY = destroy;
    }

    private Bnns() {
    }

    /** True on macOS with BNNS available. */
    public static boolean available() {
        return CREATE != null;
    }

    /** One filter: the shape's convolution with the weights (and bias) of one version. */
    private static final class Filter {
        final MemorySegment handle;
        final Object owner;
        final long version;
        final Arena arena = Arena.ofShared();
        /** Descriptors of dx, dy and dw for the backward pass (their data set per call). */
        final MemorySegment inDelta;
        final MemorySegment outDelta;
        final MemorySegment weightsDelta;

        Filter(MemorySegment handle, Object owner, long version, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int groups) {
            this.handle = handle;
            this.owner = owner;
            this.version = version;
            inDelta = nd(arena, LAYOUT_CHW, MemorySegment.NULL, wd, h, cin);
            outDelta = nd(arena, LAYOUT_CHW, MemorySegment.NULL, ow, oh, cout);
            weightsDelta = nd(arena, LAYOUT_OIHW, MemorySegment.NULL, r, r, cin / groups, cout);
        }
    }

    private static final Map<String, Filter> FILTERS = new ConcurrentHashMap<>();

    /**
     * The filter of {@code key} for the weight {@code w} at {@code version} of {@code owner} (null
     * owner: the weight may change at any call, so a new filter each time).
     */
    private static Filter filter(String key, MemorySegment w, Object owner, long version, MemorySegment bias, boolean relu, int n, int cin, int h, int wd, int cout, int oh,
            int ow, int r, int stride, int pad, int dil, int groups, boolean backward) {
        Filter f = FILTERS.get(key);
        if (f != null && owner != null && f.owner == owner && f.version == version) {
            return f;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment params = arena.allocate(CONV_SIZE, 8);
            MemorySegment.copy(nd(arena, LAYOUT_CHW, MemorySegment.NULL, wd, h, cin), 0, params, 0, ND_SIZE);
            MemorySegment.copy(nd(arena, LAYOUT_OIHW, w, r, r, cin / groups, cout), 0, params, CONV_W, ND_SIZE);
            MemorySegment.copy(nd(arena, LAYOUT_CHW, MemorySegment.NULL, ow, oh, cout), 0, params, CONV_O, ND_SIZE);
            if (bias != null) {
                MemorySegment.copy(nd(arena, LAYOUT_VECTOR, bias, cout), 0, params, CONV_BIAS, ND_SIZE);
            }
            if (relu) {
                params.set(I, CONV_ACTIVATION, RELU);
            }
            long[] tail = { stride, stride, dil, dil, pad, pad, groups };
            for (int i = 0; i < tail.length; i++) {
                params.set(J, CONV_X_STRIDE + 8L * i, tail[i]);
            }
            MemorySegment fp = arena.allocate(FILTER_PARAMS_SIZE, 8);
            fp.set(J, 8, Cpu.threads());
            if (backward) {
                // BNNSFlagsUseClientPtr: required for the backward pass; the filter reads the weight where it is
                fp.set(I, 0, 1);
            }
            MemorySegment handle = (MemorySegment) CREATE.invokeExact(params, fp);
            if (handle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("BNNS cannot create the convolution " + key);
            }
            Filter made = new Filter(handle, owner, version, n, cin, h, wd, cout, oh, ow, r, groups);
            Filter old = FILTERS.put(key, made);
            if (old != null) {
                synchronized (old) {
                    DESTROY.invokeExact(old.handle);
                }
            }
            return made;
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new IllegalStateException(e);
        }
    }

    /** An N-dimensional array descriptor of FP32 values, sizes innermost first, contiguous. */
    private static MemorySegment nd(Arena arena, int layout, MemorySegment data, long... sizes) {
        MemorySegment d = arena.allocate(ND_SIZE, 8);
        d.set(I, ND_LAYOUT, layout);
        for (int i = 0; i < sizes.length; i++) {
            d.set(J, ND_SIZES + 8L * i, sizes[i]);
        }
        d.set(A, ND_DATA, data);
        d.set(I, ND_TYPE, FLOAT32);
        return d;
    }

    private static String shape(int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
        return n + "x" + cin + "x" + h + "x" + wd + " " + cout + "x" + oh + "x" + ow + " r" + r + " s" + stride + " p" + pad + " d" + dil + " g" + groups;
    }

    /** y = act(conv2d(x, w) + bias) in NCHW; {@code owner} and {@code version} identify the weight and bias. */
    public static void conv2d(MemorySegment x, MemorySegment w, Object owner, long version, MemorySegment bias, MemorySegment y, int n, int cin, int h, int wd, int cout, int oh, int ow,
            int r, int stride, int pad, int dil, int groups, boolean relu) {
        String key = "fwd " + shape(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups) + (bias != null ? " bias" : "") + (relu ? " relu" : "");
        Filter f = filter(key, w, owner, version, bias, relu, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, false);
        synchronized (f) {
            try {
                check((int) APPLY_BATCH.invokeExact(f.handle, (long) n, x, (long) cin * h * wd, y, (long) cout * oh * ow), key);
            } catch (RuntimeException e) {
                throw e;
            } catch (Throwable e) {
                throw new IllegalStateException(e);
            }
        }
    }

    /** dx = the input gradient of conv2d, from dy and the weight (NCHW). */
    public static void conv2dBackwardData(MemorySegment dy, MemorySegment w, Object owner, long version, MemorySegment dx, int n, int cin, int h, int wd, int cout, int oh, int ow, int r,
            int stride, int pad, int dil, int groups) {
        String key = "bwd " + shape(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups);
        Filter f = filter(key, w, owner, version, null, false, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, true);
        backward(f, key, n, MemorySegment.NULL, dx, dy, MemorySegment.NULL, (long) cin * h * wd, (long) cout * oh * ow);
    }

    private static final Map<String, MemorySegment> NO_WEIGHTS = new ConcurrentHashMap<>();

    /** dw = the weight gradient of conv2d, from dy and the input (NCHW). */
    public static void conv2dBackwardWeights(MemorySegment dy, MemorySegment x, MemorySegment dw, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad,
            int dil, int groups) {
        String key = "bwd_w " + shape(n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups);
        // the filter's weights do not enter the weight gradient: any values do, made once
        MemorySegment zeros = NO_WEIGHTS.computeIfAbsent(key, k -> Arena.global().allocate((long) cout * (cin / groups) * r * r * 4, 64));
        Filter f = filter(key, zeros, zeros, 0, null, false, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, true);
        backward(f, key, n, x, MemorySegment.NULL, dy, dw, (long) cin * h * wd, (long) cout * oh * ow);
    }

    private static void backward(Filter f, String key, int n, MemorySegment x, MemorySegment dx, MemorySegment dy, MemorySegment dw, long inStride, long outStride) {
        synchronized (f) {
            try {
                f.inDelta.set(A, ND_DATA, dx);
                f.outDelta.set(A, ND_DATA, dy);
                f.weightsDelta.set(A, ND_DATA, dw);
                MemorySegment inDelta = dx.equals(MemorySegment.NULL) ? MemorySegment.NULL : f.inDelta;
                MemorySegment weightsDelta = dw.equals(MemorySegment.NULL) ? MemorySegment.NULL : f.weightsDelta;
                check((int) APPLY_BACKWARD_BATCH.invokeExact(f.handle, (long) n, x, inStride, inDelta, inStride, MemorySegment.NULL, outStride, f.outDelta, outStride, weightsDelta,
                        MemorySegment.NULL), key);
            } catch (RuntimeException e) {
                throw e;
            } catch (Throwable e) {
                throw new IllegalStateException(e);
            }
        }
    }

    private static void check(int status, String what) {
        if (status != 0) {
            throw new IllegalStateException("BNNS " + what + " failed with status " + status);
        }
    }
}
