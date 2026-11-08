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

import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.JitnnConfig;

/**
 * Vendor libraries that some operators can call instead of jitNN's own kernels: cuBLAS, cuDNN,
 * cuFFT and cuSPARSE on CUDA, MLX on Metal (through TornadoVM's library tasks, inside the same
 * task graph as the JIT kernels).
 *
 * <p>Off by default. {@code -Djitnn.native=auto} (or {@code -Djitnn.withNative=true}) uses every
 * library that is available on the plan's device; a comma-separated list ({@code cublas,cudnn,cufft,cusparse,cutlass,mlx}) restricts it.
 * A library is available when TornadoVM's module for it is loaded and a one-task probe plan runs on
 * the device; one that is missing, or that fails later, is left to the JIT kernels.
 */
public final class Native {

    /** The libraries jitNN can call. */
    public enum Lib {
        /** NVIDIA cuBLAS (CUDA): matrix products. */
        CUBLAS(TornadoVMBackendType.CUDA),
        /** NVIDIA cuDNN (CUDA): convolutions, softmax and attention. */
        CUDNN(TornadoVMBackendType.CUDA),
        /** NVIDIA cuFFT (CUDA): Fourier transforms. */
        CUFFT(TornadoVMBackendType.CUDA),
        /** NVIDIA cuSPARSE (CUDA): sparse products. */
        CUSPARSE(TornadoVMBackendType.CUDA),
        /** NVIDIA CUTLASS (CUDA): matrix products. */
        CUTLASS(TornadoVMBackendType.CUDA),
        /** NVIDIA NCCL (CUDA): collectives across GPUs. */
        NCCL(TornadoVMBackendType.CUDA),
        /** Apple MLX (Metal): matrix products, convolutions, normalisation, softmax and attention. */
        MLX(TornadoVMBackendType.METAL);

        final TornadoVMBackendType backend;

        Lib(TornadoVMBackendType backend) {
            this.backend = backend;
        }
    }

    private static final Set<Lib> WANTED = parse();
    /** Whether the libraries were requested as a whole ({@code auto}) rather than by name. */
    private static final boolean AUTO = java.util.List.of("auto", "on", "true", "all").contains(mode());

    static {
        // MLX calls commit without waiting for the GPU unless -Dtornado.mlx.async=false: waiting after
        // each call made batch-1 BERT 3.8x slower on an M4 Pro (TornadoVM waits before host reads)
        if (WANTED.contains(Lib.MLX) && System.getProperty("tornado.mlx.async") == null) {
            System.setProperty("tornado.mlx.async", "true");
        }
        // Device arrays start 128-byte aligned unless -Dtornado.cuda.payloadAlignment says otherwise:
        // TornadoVM's default of 32 makes cuBLAS take slower tensor-core kernels (BERT FP16 11-17%
        // slower on an RTX 5070 Ti); jitNN's own kernels run the same at either
        if (WANTED.stream().anyMatch(l -> l.backend == TornadoVMBackendType.CUDA) && System.getProperty("tornado.cuda.payloadAlignment") == null) {
            System.setProperty("tornado.cuda.payloadAlignment", "128");
        }
    }
    private static final boolean VERBOSE = JitnnConfig.current().get(JitnnConfig.VERBOSE) || JitnnConfig.current().get(JitnnConfig.NATIVE_VERBOSE);
    /** Per device key and library: available (probed) or not. */
    private static final Map<String, Map<Lib, Boolean>> AVAILABLE = new java.util.HashMap<>();
    /** Library functions ("cudnn.sdpa") turned off at run time: a plan that called them failed. */
    private static final Set<String> DISABLED = new java.util.TreeSet<>();
    /** Library functions that have run in some plan: a later failure is not theirs. */
    private static final Set<String> PROVEN = new java.util.HashSet<>();
    /** Libraries treated as missing (tests of the fallback): {@code -Djitnn.native.unavailable=cudnn,...}. */
    private static final String UNAVAILABLE = JitnnConfig.current().get(JitnnConfig.NATIVE_UNAVAILABLE).toLowerCase(Locale.ROOT);

    private Native() {
    }

    private static String mode() {
        return java.util.Objects.requireNonNullElse(JitnnConfig.current().get(JitnnConfig.NATIVE), JitnnConfig.current().get(JitnnConfig.WITH_NATIVE) ? "auto" : "off").trim().toLowerCase(Locale.ROOT);
    }

    private static Set<Lib> parse() {
        String mode = mode();
        Set<Lib> libs = EnumSet.noneOf(Lib.class);
        switch (mode) {
            case "", "off", "false", "none" -> {
            }
            case "auto", "on", "true", "all" -> libs.addAll(EnumSet.allOf(Lib.class));
            default -> {
                for (String s : mode.split(",")) {
                    libs.add(Lib.valueOf(s.trim().toUpperCase(Locale.ROOT)));
                }
            }
        }
        return libs;
    }

    /** True when the libraries were requested as a whole ({@code auto}), not by name. */
    public static boolean auto() {
        return AUTO;
    }

    /** True when any library is requested (the option is on). */
    public static boolean requested() {
        return !WANTED.isEmpty();
    }

    /**
     * Whether operators on the current plan's device should call {@code lib}: requested, for this
     * backend, not disabled, and available (probed once per device).
     */
    public static synchronized boolean use(String function, Lib lib) {
        if (Engine.backend() != Backend.TORNADO || !WANTED.contains(lib) || DISABLED.contains(feature(lib, function)) || Devices.backend() != lib.backend) {
            return false;
        }
        if (!use(lib)) {
            return false;
        }
        // functions with a probe of their own (cuDNN's fused attention builds no plan on some GPUs)
        String key = Devices.key() + "/" + feature(lib, function);
        Boolean fn = FUNCTIONS.get(key);
        if (fn == null) {
            String why;
            try {
                why = LibraryProbe.get().probeFunction(lib, function);
            } catch (LinkageError | RuntimeException ex) {
                why = ex.getClass().getSimpleName() + ": " + firstLine(ex);
            }
            fn = why == null;
            if (why != null) {
                System.err.printf("[jitnn] native %s: not available on %s (%s); using jitNN kernels%n", feature(lib, function), Devices.name(), why);
            }
            FUNCTIONS.put(key, fn);
        }
        return fn;
    }

    /** Per device and library function: whether its own probe ran (functions without one count as available). */
    private static final Map<String, Boolean> FUNCTIONS = new java.util.HashMap<>();

    private static synchronized boolean use(Lib lib) {
        Map<Lib, Boolean> perDevice = AVAILABLE.computeIfAbsent(Devices.key(), k -> new EnumMap<>(Lib.class));
        Boolean ok = perDevice.get(lib);
        if (ok == null) {
            ok = probe(lib);
            perDevice.put(lib, ok);
        }
        return ok;
    }

    /** "cudnn.conv": a library function, the unit that is turned off when it fails. */
    static String feature(Lib lib, String function) {
        return lib.name().toLowerCase(Locale.ROOT) + "." + function;
    }

    private static boolean probe(Lib lib) {
        String why;
        try {
            why = java.util.Arrays.asList(UNAVAILABLE.split(",")).contains(lib.name().toLowerCase(Locale.ROOT)) ? "treated as missing (jitnn.native.unavailable)"
                    : LibraryProbe.get().probe(lib);
        } catch (LinkageError | RuntimeException ex) {
            // the TornadoVM module is not loaded, or the probe plan failed
            why = ex.getClass().getSimpleName() + ": " + firstLine(ex);
        }
        if (why == null) {
            if (VERBOSE) {
                System.err.printf("[jitnn] native %s: available on %s%n", lib.name().toLowerCase(Locale.ROOT), Devices.name());
            }
            return true;
        }
        System.err.printf("[jitnn] native %s: not available on %s (%s); using jitNN kernels%n", lib.name().toLowerCase(Locale.ROOT), Devices.name(), why);
        return false;
    }

    private static String firstLine(Throwable t) {
        Throwable c = t;
        String msg = t.getMessage();
        while (c.getCause() != null && c.getCause() != c) {
            c = c.getCause();
            if (c.getMessage() != null) {
                msg = c.getMessage();
            }
        }
        return msg == null ? "" : msg.lines().findFirst().orElse("");
    }

    /**
     * A plan calling {@code features} failed on its first execution: stops using those of them that
     * have not run successfully elsewhere (all of them if every one has); later plans use jitNN kernels.
     */
    static synchronized void disable(Set<String> features, Throwable cause) {
        Set<String> suspects = new java.util.TreeSet<>(features);
        suspects.removeAll(PROVEN);
        if (suspects.isEmpty()) {
            suspects.addAll(features);
        }
        DISABLED.addAll(suspects);
        System.err.printf("[jitnn] native %s failed (%s); using jitNN kernels instead%n", suspects, firstLine(cause));
    }

    /** A plan calling {@code features} ran: they work on this device. */
    static synchronized void proven(Set<String> features) {
        PROVEN.addAll(features);
    }

    /** A task graph on the current plan's device, for a probe (see {@link #runProbe}). */
    public static uk.ac.manchester.tornado.api.TaskGraph probeGraph(String name) {
        String graph = "jitnnProbe" + name;
        Devices.pin(graph);
        return new uk.ac.manchester.tornado.api.TaskGraph(graph);
    }

    /** Runs a probe graph once on the current plan's device. */
    public static void runProbe(uk.ac.manchester.tornado.api.TaskGraph graph) {
        try (uk.ac.manchester.tornado.api.TornadoExecutionPlan plan = new uk.ac.manchester.tornado.api.TornadoExecutionPlan(graph.snapshot())) {
            Devices.apply(plan);
            plan.execute();
        } catch (uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException e) {
            throw new ai.beestream.jitnn.JitnnException("a vendor library probe failed on " + Devices.name(), e);
        }
    }

    private static final Map<Lib, Long> EMITTED = new EnumMap<>(Lib.class);

    static synchronized void emitted(Lib lib) {
        EMITTED.merge(lib, 1L, Long::sum);
    }

    /** Library tasks compiled into plans so far, per library (calls that replays repeat are not counted). */
    public static synchronized Map<Lib, Long> emittedTasks() {
        return new EnumMap<>(EMITTED);
    }

    /** Part of the plan cache key: the libraries in use, as they change what operators emit. */
    static synchronized String keyTag() {
        if (WANTED.isEmpty()) {
            return "";
        }
        return "N" + DISABLED + "|";
    }
}
