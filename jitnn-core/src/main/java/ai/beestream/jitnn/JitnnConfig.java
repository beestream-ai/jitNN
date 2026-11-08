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
package ai.beestream.jitnn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.function.Function;

/**
 * jitNN's options, in one immutable object. Each option has a key ({@code jitnn.*}), a type, a
 * default and a description; {@code docs/CONFIGURATION.md} lists them all.
 *
 * <p>The runtime reads its configuration once, the first time it needs an option, and keeps it:
 * by default the {@code jitnn.*} system properties at that moment ({@link #fromSystemProperties}).
 * To configure jitNN from code, {@link #install} a configuration before the first tensor
 * operation:
 *
 * <pre>{@code
 * JitnnConfig.install(JitnnConfig.builder()
 *         .set(JitnnConfig.PLAN_CACHE, 64)
 *         .set(JitnnConfig.NATIVE, "auto")
 *         .build());
 * }</pre>
 *
 * <p>Only the options marked {@linkplain Option#changeableAtRunTime() changeable at run time}
 * (determinism, autotuning, the GEMM and transpose variants, the CPU backend's library choice)
 * can change afterwards, through {@link #change} or their setters such as
 * {@code JitnnRuntime.setDeterministic}; the others are start-time options.
 *
 * <p>An invalid value (a number that does not parse, a value outside an option's choices) fails
 * with an {@link IllegalArgumentException} naming the key.
 */
public final class JitnnConfig {

    /** The groups options are listed in. */
    public enum Section {
        /** Execution, devices, plans and memory. */
        RUNTIME("Runtime options"),
        /** jitNN's kernel choices and graph rewrites. */
        KERNELS("Kernel variants"),
        /** The ONNX, TensorFlow and DJL modules. */
        MODELS("Imported models and DJL"),
        /** Vendor libraries (cuBLAS, cuDNN, CUTLASS, MLX and others). */
        NATIVE("Vendor libraries");

        private final String title;

        Section(String title) {
            this.title = title;
        }

        /** The heading this section has in the documentation. */
        public String title() {
            return title;
        }
    }

    /**
     * One option: its key, type, default and description.
     *
     * @param <T> the value type
     */
    public static final class Option<T> {
        private final String key;
        private final Class<T> type;
        private final T defaultValue;
        private final Section section;
        private final String defaultText;
        private final List<String> choices;
        private final String description;
        private final Function<String, T> parser;
        private boolean runtime;

        private Option(String key, Class<T> type, T defaultValue, Section section, String defaultText, List<String> choices, String description, Function<String, T> parser) {
            this.key = key;
            this.type = type;
            this.defaultValue = defaultValue;
            this.section = section;
            this.defaultText = defaultText;
            this.choices = choices;
            this.description = description;
            this.parser = parser;
        }

        private Option<T> runtime() {
            runtime = true;
            return this;
        }

        /** The system property, e.g. {@code jitnn.plan.cache}. */
        public String key() {
            return key;
        }

        /** The value type: Boolean, Integer, Long, Double or String. */
        public Class<T> type() {
            return type;
        }

        /** The default, or null when it depends on the machine or on other options. */
        public T defaultValue() {
            return defaultValue;
        }

        /** The section the documentation lists this option in. */
        public Section section() {
            return section;
        }

        /** The default as the documentation states it (Markdown). */
        public String defaultText() {
            return defaultText;
        }

        /** The accepted values of an option with a fixed set of them, else an empty list. */
        public List<String> choices() {
            return choices;
        }

        /** What the option does (Markdown). */
        public String description() {
            return description;
        }

        /** Whether the option may change after the runtime has started ({@link JitnnConfig#change}). */
        public boolean changeableAtRunTime() {
            return runtime;
        }

        /** Parses a property value, failing with a message that names the key. */
        public T parse(String value) {
            try {
                return parser.apply(value.trim());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(key + ": " + e.getMessage(), e);
            }
        }

        @Override
        public String toString() {
            return key;
        }
    }

    private static final List<Option<?>> OPTIONS = new ArrayList<>();

    private static <T> Option<T> add(Option<T> o) {
        OPTIONS.add(o);
        return o;
    }

    private static Option<Boolean> bool(String key, Boolean def, Section s, String defText, String doc) {
        return add(new Option<>(key, Boolean.class, def, s, defText, List.of(), doc, v -> switch (v.toLowerCase(Locale.ROOT)) {
            case "true" -> true;
            case "false" -> false;
            default -> throw new IllegalArgumentException("expected true or false, not \"" + v + "\"");
        }));
    }

    private static Option<Integer> integer(String key, Integer def, Section s, String defText, String doc) {
        return add(new Option<>(key, Integer.class, def, s, defText, List.of(), doc, v -> {
            try {
                return Integer.valueOf(v);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("expected an integer, not \"" + v + "\"");
            }
        }));
    }

    private static Option<Long> longValue(String key, Long def, Section s, String defText, String doc) {
        return add(new Option<>(key, Long.class, def, s, defText, List.of(), doc, v -> {
            try {
                return Long.valueOf(v);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("expected an integer, not \"" + v + "\"");
            }
        }));
    }

    private static Option<Double> real(String key, Double def, Section s, String defText, String doc) {
        return add(new Option<>(key, Double.class, def, s, defText, List.of(), doc, v -> {
            try {
                return Double.valueOf(v);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("expected a number, not \"" + v + "\"");
            }
        }));
    }

    private static Option<String> string(String key, String def, Section s, String defText, String doc) {
        return add(new Option<>(key, String.class, def, s, defText, List.of(), doc, v -> v));
    }

    private static Option<String> choice(String key, String def, Section s, String defText, String[] values, String doc) {
        List<String> list = List.of(values);
        return add(new Option<>(key, String.class, def, s, defText, list, doc, v -> {
            if (!list.contains(v)) {
                throw new IllegalArgumentException("expected one of " + String.join(", ", list) + ", not \"" + v + "\"");
            }
            return v;
        }));
    }

    private static final Section RUNTIME = Section.RUNTIME;
    private static final Section KERNELS = Section.KERNELS;
    private static final Section MODELS = Section.MODELS;
    private static final Section NATIVE_LIBS = Section.NATIVE;

    /** {@code jitnn.backend}. */
    public static final Option<String> BACKEND = choice("jitnn.backend", null, RUNTIME, "`tornado` with a TornadoVM runtime, else `cpu` with `--add-modules jdk.incubator.vector`, else `reference`", new String[] { "tornado", "cpu", "reference" }, "where operators run: `tornado` (a TornadoVM device), `cpu` (the host, multithreaded, Java Vector API; operators without a CPU kernel run their reference) or `reference` (the host, scalar, one operator at a time)");

    /** {@code jitnn.device}. */
    public static final Option<String> DEVICE = string("jitnn.device", "", RUNTIME, "TornadoVM's default device", "`<driver>:<device>`, as listed by `tornado --devices`");

    /** {@code jitnn.concurrency}. */
    public static final Option<Integer> CONCURRENCY = integer("jitnn.concurrency", 1, RUNTIME, "`1`", "concurrent execution slots, each with its own plans and device buffers (weights included); threads are spread over them. Plans on different devices always run concurrently. `jitnn.plan.cache` applies per slot");

    /** {@code jitnn.cpu.threads}. */
    public static final Option<Integer> CPU_THREADS = integer("jitnn.cpu.threads", null, RUNTIME, "the physical cores this process may use, at most one socket's; on Apple silicon the performance cores", "threads of the `cpu` backend's parallel loops, the calling thread included");

    /** {@code jitnn.cpu.spinMicros}. */
    public static final Option<Long> CPU_SPIN_MICROS = longValue("jitnn.cpu.spinMicros", null, RUNTIME, "`200` (`0` on macOS)", "how long idle `cpu` backend threads (and a waiting caller) spin before sleeping");

    /** {@code jitnn.cpu.poolBytes}. */
    public static final Option<Long> CPU_POOL_BYTES = longValue("jitnn.cpu.poolBytes", 2L << 30, RUNTIME, "`2147483648` (2 GiB)", "host arrays the `cpu` backend keeps for reuse, unless `jitnn.hostPool.bytes` is set");

    /** {@code jitnn.cpu.gemm.kc}. */
    public static final Option<Integer> CPU_GEMM_KC = integer("jitnn.cpu.gemm.kc", 256, RUNTIME, "`256`", "`cpu` GEMM blocking: depth of A and B per block");

    /** {@code jitnn.cpu.gemm.mcPanels}. */
    public static final Option<Integer> CPU_GEMM_MC_PANELS = integer("jitnn.cpu.gemm.mcPanels", 16, RUNTIME, "`16`", "`cpu` GEMM blocking: rows of A per tile, in panels of 6");

    /** {@code jitnn.cpu.gemm.nc}. */
    public static final Option<Integer> CPU_GEMM_NC = integer("jitnn.cpu.gemm.nc", 256, RUNTIME, "`256`", "`cpu` GEMM blocking: columns of B per tile");

    /** {@code jitnn.cpu.gemm.tileMacs}. */
    public static final Option<Long> CPU_GEMM_TILE_MACS = longValue("jitnn.cpu.gemm.tileMacs", 1L << 20, RUNTIME, "`1048576`", "`cpu` GEMM: fewest multiply-adds per parallel tile");

    /** {@code jitnn.cpu.opencl}. */
    public static final Option<String> CPU_OPENCL = string("jitnn.cpu.opencl", "auto", RUNTIME, "`auto`", "where the `cpu` backend runs operators it has no kernel for. `auto`: the first OpenCL CPU device (PoCL, Intel CPU runtime) when a TornadoVM runtime is present and the JVM runs with `LD_PRELOAD=$JAVA_HOME/lib/libjsig.so` (without it the drivers' signal handlers crash the JVM). Give the driver the backend's thread count: `POCL_MAX_PTHREAD_COUNT`, or `POCL_CPU_MAX_CU_COUNT` after PoCL 3; `CL_CONFIG_CPU_TBB_NUM_WORKERS` for Intel. `off`: the reference. `<driver>:<device>`: that device. Each operator and shape is timed on the reference and the device and keeps the faster; consecutive device operators run as one plan; one the device fails to compile (FP16 without `cl_khr_fp16`) stays on the reference");

    /** {@code jitnn.cpu.opencl.choose}. */
    public static final Option<String> CPU_OPENCL_CHOOSE = choice("jitnn.cpu.opencl.choose", "measure", RUNTIME, "`measure`", new String[] { "measure", "always" }, "`always`: send operators without a CPU kernel to the `jitnn.cpu.opencl` device without timing (tests)").runtime();

    /** {@code jitnn.cpu.blas}. */
    public static final Option<String> CPU_BLAS = string("jitnn.cpu.blas", "auto", RUNTIME, "`auto`", "native BLAS (`cblas_sgemm`) for the `cpu` backend's linear, matmul and convolutions, forward and backward: `auto` tries MKL, OpenBLAS, Arm Performance Libraries and Accelerate on the library path; `off`; or a library file. Chosen per operator and shape by `jitnn.cpu.native.choose`");

    /** {@code jitnn.cpu.onednn}. */
    public static final Option<String> CPU_ONEDNN = string("jitnn.cpu.onednn", "auto", RUNTIME, "`auto`", "oneDNN (`libdnnl`) for the `cpu` backend's convolutions (forward with bias and ReLU, both gradients, grouped): `auto`, `off` or a library file. Use an OpenMP build, whose thread count jitNN sets; a TBB build takes every core");

    /** {@code jitnn.cpu.bnns}. */
    public static final Option<String> CPU_BNNS = string("jitnn.cpu.bnns", "auto", RUNTIME, "`auto`", "on macOS, BNNS (Accelerate) for the same convolutions; `off` never");

    /** {@code jitnn.cpu.native.choose}. */
    public static final Option<String> CPU_NATIVE_CHOOSE = choice("jitnn.cpu.native.choose", "measure", RUNTIME, "`measure`", new String[] { "measure", "kernel", "blas", "onednn", "bnns" }, "how the `cpu` backend picks between its kernel and `blas`, `onednn`, `bnns` per operator and shape: `measure` times each way for three calls, runs the fastest, and re-times after 64 more calls for the final choice; `kernel`, `blas`, `onednn` or `bnns` forces that way when available (tests)").runtime();

    /** {@code jitnn.cpu.reuse}. */
    public static final Option<Boolean> CPU_REUSE = bool("jitnn.cpu.reuse", true, RUNTIME, "`true`", "a `cpu` trace of the same structure as the last reuses its intermediates' arrays, as a compiled plan does; tensors not requested at that sync (`JitnnRuntime.sync(t)`, `t.item()`) are then gone");

    /** {@code jitnn.cpu.reuse.traces}. */
    public static final Option<Integer> CPU_REUSE_TRACES = integer("jitnn.cpu.reuse.traces", 32, RUNTIME, "`32`", "trace structures each thread remembers for that reuse");

    /** {@code jitnn.cpu.stats}. */
    public static final Option<Boolean> CPU_STATS = bool("jitnn.cpu.stats", false, RUNTIME, "`false`", "time every `cpu` operator and print calls, time and reference fallbacks at exit; `JitnnRuntime.cpuStats()` gives counts without it");

    /** {@code jitnn.cpu.stats.shapes}. */
    public static final Option<Boolean> CPU_STATS_SHAPES = bool("jitnn.cpu.stats.shapes", false, RUNTIME, "`false`", "with `jitnn.cpu.stats`: one row per operator and shape");

    /** {@code jitnn.plan.cache}. */
    public static final Option<Integer> PLAN_CACHE = integer("jitnn.plan.cache", 16, RUNTIME, "`16` (`64` under `jitnn-djl` when not set)", "compiled plans kept; the least recently used is closed, saving parameters first");

    /** {@code jitnn.plan.idleBytes}. */
    public static final Option<Long> PLAN_IDLE_BYTES = longValue("jitnn.plan.idleBytes", 64L << 20, RUNTIME, "`67108864` (64 MiB)", "a plan holding at least this much intermediate memory is closed after `jitnn.plan.idleExecutions` executions without it, and recompiled if needed again");

    /** {@code jitnn.plan.idleExecutions}. */
    public static final Option<Long> PLAN_IDLE_EXECUTIONS = longValue("jitnn.plan.idleExecutions", 256L, RUNTIME, "`256`", "executions of other plans before an idle large plan is closed (checked every 32)");

    /** {@code jitnn.plan.carry}. */
    public static final Option<Boolean> PLAN_CARRY = bool("jitnn.plan.carry", true, RUNTIME, "`true`", "an input that is an output of the same plan's previous execution (a gradient accumulated without `zeroGrad`) is copied on the device instead of round-tripping through the host; the plan recompiles once when this is first seen");

    /** {@code jitnn.plan.memo}. */
    public static final Option<Boolean> PLAN_MEMO = bool("jitnn.plan.memo", true, RUNTIME, "`true`", "a trace recorded again with the same operations, shapes and weights replays its last plan without rewriting the trace or building its key, once that plan has run twice; not for traces with a large host input");

    /** {@code jitnn.promoteInputBytes}. */
    public static final Option<Long> PROMOTE_INPUT_BYTES = longValue("jitnn.promoteInputBytes", 16L << 20, RUNTIME, "`16777216` (16 MiB)", "a host input at least this large that a later execution reads again becomes persistent device storage");

    /** {@code jitnn.cudaGraph}. */
    public static final Option<Boolean> CUDA_GRAPH = bool("jitnn.cudaGraph", true, RUNTIME, "`true`", "on CUDA, capture each plan as a CUDA graph and replay it");

    /** {@code jitnn.dynamicShapes}. */
    public static final Option<Boolean> DYNAMIC_SHAPES = bool("jitnn.dynamicShapes", true, RUNTIME, "`true`", "share kernels across array sizes (TornadoVM `withDynamicShapes()`), so a new batch or sequence size reuses compiled kernels");

    /** {@code jitnn.graph.maxTasks}. */
    public static final Option<Integer> GRAPH_MAX_TASKS = integer("jitnn.graph.maxTasks", 40, RUNTIME, "`40`", "tasks per TornadoVM task graph; longer traces become a chain of graphs (a graph holds at most 256 tasks and 8 KiB of bytecode)");

    /** {@code jitnn.hostPool.bytes}. */
    public static final Option<Long> HOST_POOL_BYTES = longValue("jitnn.hostPool.bytes", 256L << 20, RUNTIME, "`268435456` (256 MiB)", "host arrays kept for reuse: outputs that outlive their plan's next run are copied into one, returned on `Tensor.release` or collection");

    /** {@code jitnn.parallelCopyBytes}. */
    public static final Option<Long> PARALLEL_COPY_BYTES = longValue("jitnn.parallelCopyBytes", 32L << 20, RUNTIME, "`33554432` (32 MiB)", "host inputs at least this large are copied into a plan's buffers by up to eight threads");

    /** {@code jitnn.reuseBuffers}. */
    public static final Option<Boolean> REUSE_BUFFERS = bool("jitnn.reuseBuffers", true, RUNTIME, "`true`", "reuse dead backward-pass and inference buffers inside a compiled step");

    /** {@code jitnn.reuseBuffers.exact}. */
    public static final Option<String> REUSE_BUFFERS_EXACT = choice("jitnn.reuseBuffers.exact", "auto", RUNTIME, "`auto`", new String[] { "auto", "true", "false" }, "reuse only dead buffers of exactly the same size, so vendor-library operands need no copies: `auto` (on Metal with MLX), `true`, `false`");

    /** {@code jitnn.verbose}. */
    public static final Option<Boolean> VERBOSE = bool("jitnn.verbose", false, RUNTIME, "`false`", "log every compiled plan (device, nodes, tasks, build time) and the vendor libraries found");

    /** {@code jitnn.debugKeys}. */
    public static final Option<Boolean> DEBUG_KEYS = bool("jitnn.debugKeys", false, RUNTIME, "`false`", "print each new plan's cache key to standard error, to find what makes a trace recompile");

    /** {@code jitnn.printKernels}. */
    public static final Option<String> PRINT_KERNELS = string("jitnn.printKernels", "", RUNTIME, "`false`", "print each compiled plan's operators and kernel source: `true` for standard output, or a file name (appended); see [Printing the kernels](USAGE.md#printing-the-kernels)");

    /** {@code jitnn.profile}. */
    public static final Option<Boolean> PROFILE = bool("jitnn.profile", false, RUNTIME, "`false`", "TornadoVM's profiler: `JitnnRuntime.lastProfile()` gives the last step's kernel time, `JitnnRuntime.kernelBreakdown()` device time per operator task. `JitnnRuntime.startTrace`/`stopTrace` write a Chrome trace of plan runs and kernels. On CUDA also set `jitnn.cudaGraph=false`: the profiler reports no kernel time under CUDA graphs");

    /** {@code jitnn.gemm}. */
    public static final Option<String> GEMM = string("jitnn.gemm", "auto", KERNELS, "`auto`", "FP32 GEMM kernel (see [GEMM selection](#gemm-selection-auto)): `auto`, `simdgroup`, `128`, `64`, `16`, or padded `gemm64p`, `gemm64pd`, `gemm128p`, `gemm128pd`, `gemm128x64pd`, `gemm64x128pd`. A forced variant also disables forward split-K").runtime();

    /** {@code jitnn.tuning}. */
    public static final Option<String> TUNING = string("jitnn.tuning", "", KERNELS, "empty", "GEMM tuning table: empty reads `tuning/<backend>-<device>.json` from the class path (tables ship for the A10 and A100 80GB PCIe), a path reads that file, `off` ignores tables. Used by `jitnn.gemm=auto` on CUDA and OpenCL");

    /** {@code jitnn.fuse.program}. */
    public static final Option<Boolean> FUSE_PROGRAM = bool("jitnn.fuse.program", true, KERNELS, "`true`", "fused element-wise chains of up to six operations sharing the output's layout use a kernel specialised for the chain (`FusedKernels.fusedProgram1` to `6`); `false` uses the generic interpreter kernel");

    /** {@code jitnn.deterministic}. */
    public static final Option<Boolean> DETERMINISTIC = bool("jitnn.deterministic", false, KERNELS, "`false`", "reproducible runs: no autotuning measurements, deterministic cuDNN algorithms (`JitnnRuntime.setDeterministic`)").runtime();

    /** {@code jitnn.autotune}. */
    public static final Option<String> AUTOTUNE = choice("jitnn.autotune", "off", KERNELS, "`off`", new String[] { "off", "on", "use" }, "run-time autotuning: `on` measures the variants of each new GEMM, 1x1 and 3x3 convolution and row-norm shape on first use and keeps the fastest; `use` only applies saved decisions. Decisions override tuning tables and size rules").runtime();

    /** {@code jitnn.autotune.dir}. */
    public static final Option<String> AUTOTUNE_DIR = string("jitnn.autotune.dir", null, KERNELS, "`~/.jitnn/autotune`", "where decisions are saved, one `<backend>-<device>.properties` per device").runtime();

    /** {@code jitnn.autotune.margin}. */
    public static final Option<Double> AUTOTUNE_MARGIN = real("jitnn.autotune.margin", 0.05, KERNELS, "`0.05`", "how much faster a variant must be to replace the rule's choice");

    /** {@code jitnn.autotune.reps}. */
    public static final Option<Integer> AUTOTUNE_REPS = integer("jitnn.autotune.reps", 8, KERNELS, "`8`", "operator copies per autotuning measurement plan");

    /** {@code jitnn.autotune.runs}. */
    public static final Option<Integer> AUTOTUNE_RUNS = integer("jitnn.autotune.runs", 15, KERNELS, "`15`", "timed executions of a measurement plan (the median is used)");

    /** {@code jitnn.autotune.verbose}. */
    public static final Option<Boolean> AUTOTUNE_VERBOSE = bool("jitnn.autotune.verbose", false, KERNELS, "`false`", "print each measurement and decision");

    /** {@code jitnn.splitK}. */
    public static final Option<Boolean> SPLIT_K = bool("jitnn.splitK", true, KERNELS, "`true`", "split the reduction of GEMMs and convolutions whose output fills too few work-groups (A^T B weight gradients, forward products), summing partials in order");

    /** {@code jitnn.splitK.groups}. */
    public static final Option<Integer> SPLIT_K_GROUPS = integer("jitnn.splitK.groups", 256, KERNELS, "`256`", "work-groups a forward product should reach before it stops splitting K");

    /** {@code jitnn.tensorCores}. */
    public static final Option<Boolean> TENSOR_CORES = bool("jitnn.tensorCores", true, KERNELS, "`true`", "on CUDA, FP16 and BF16 products run on tensor cores (`mma.sync` m16n8k16, compute capability 8.0+)");

    /** {@code jitnn.transpose}. */
    public static final Option<String> TRANSPOSE = choice("jitnn.transpose", "32", KERNELS, "`32`", new String[] { "32", "16", "naive" }, "transpose kernel: `32` (32x32 tiles), `16` or `naive`").runtime();

    /** {@code jitnn.binaryRepeat}. */
    public static final Option<Boolean> BINARY_REPEAT = bool("jitnn.binaryRepeat", true, KERNELS, "`true`", "a binary operator whose broadcast operand repeats one contiguous block uses an index kernel (`(i / div) % mod`) instead of the general broadcast kernel");

    /** {@code jitnn.fusedAttention}. */
    public static final Option<Boolean> FUSED_ATTENTION = bool("jitnn.fusedAttention", true, KERNELS, "`true`", "FP32 attention with head dimension up to 128 uses the fused FlashAttention-style kernels; `false` composes batched GEMMs and a softmax");

    /** {@code jitnn.flash}. */
    public static final Option<String> FLASH = choice("jitnn.flash", "tiled", KERNELS, "tiled", new String[] { "tiled", "simple" }, "`simple`: head dimensions up to 64 use the generic fused attention kernels instead of the tiled ones");

    /** {@code jitnn.flash.mma}. */
    public static final Option<Boolean> FLASH_MMA = bool("jitnn.flash.mma", true, KERNELS, "`true`", "on CUDA with `jitnn.tensorCores`, FP16 attention (forward and backward, head dimension up to 128) runs on tensor cores; `false` uses the tiled FP16 kernels up to 64 and FP32 copies above");

    /** {@code jitnn.rownorm}. */
    public static final Option<String> ROWNORM = choice("jitnn.rownorm", "warp", KERNELS, "warp", new String[] { "warp", "group" }, "`group`: LayerNorm and RMSNorm on CUDA use one work-group per row instead of one warp (used from 1024 rows of at most 768 features)");

    /** {@code jitnn.conv}. */
    public static final Option<String> CONV = choice("jitnn.conv", "padded", KERNELS, "padded", new String[] { "padded", "simple" }, "`simple`: plain implicit-GEMM convolution kernels instead of the padded, prefetching ones");

    /** {@code jitnn.winograd}. */
    public static final Option<String> WINOGRAD = choice("jitnn.winograd", "auto", KERNELS, "`auto`", new String[] { "auto", "2", "false" }, "3x3 stride-1 convolutions (one group, at least 16 channels in and out, padding up to 2) use Winograd F(4x4, 3x3), or F(2x2, 3x3) with too few 4x4 tiles, forward and both gradients. `2` keeps F(2x2, 3x3), `false` disables Winograd");

    /** {@code jitnn.conv1x1}. */
    public static final Option<String> CONV1X1 = choice("jitnn.conv1x1", "gemm", KERNELS, "GEMM", new String[] { "gemm", "implicit" }, "`implicit`: 1x1 stride-1 convolutions use the implicit-GEMM kernels instead of a batched GEMM over the images");

    /** {@code jitnn.conv1x1.gemm}. */
    public static final Option<String> CONV1X1_GEMM = choice("jitnn.conv1x1.gemm", "auto", KERNELS, "`auto`", new String[] { "auto", "64", "128x64", "64x128", "128x16" }, "1x1 convolution GEMM tile: `64`, `128x64`, `64x128`, `128x16` (for 4x4 planes), or `auto` (by output rows and plane size)");

    /** {@code jitnn.conv1x1.backward}. */
    public static final Option<String> CONV1X1_BACKWARD = choice("jitnn.conv1x1.backward", "gemm", KERNELS, "`gemm`", new String[] { "gemm", "implicit" }, "`implicit`: the 1x1 input gradient uses the implicit-GEMM kernel instead of the batched GEMM");

    /** {@code jitnn.convPhases}. */
    public static final Option<Boolean> CONV_PHASES = bool("jitnn.convPhases", true, KERNELS, "`true`", "input gradient of strided convolutions as a GEMM and a gather (when the column buffer fits in 256 MB), else one implicit GEMM per stride phase; `false` uses the general gathering kernel");

    /** {@code jitnn.conv.weightGroups}. */
    public static final Option<Integer> CONV_WEIGHT_GROUPS = integer("jitnn.conv.weightGroups", 256, KERNELS, "`256`", "the filter gradient is split over output positions until it reaches about this many work-groups");

    /** {@code jitnn.conv.weightChunk}. */
    public static final Option<Integer> CONV_WEIGHT_CHUNK = integer("jitnn.conv.weightChunk", 256, KERNELS, "`256`", "fewest output positions per slice of that split");

    /** {@code jitnn.rnn}. */
    public static final Option<String> RNN = choice("jitnn.rnn", "fused", KERNELS, "`fused`", new String[] { "fused", "steps", "simple" }, "LSTM layers of up to 512 hidden units and 256 batch rows run the whole sequence in one launch (forward and backward); `steps` launches tiled kernels per timestep, `simple` thread-per-cell kernels. GRU always runs per timestep (tiled, or thread-per-cell with `simple`)");

    /** {@code jitnn.gguf.tiledRows}. */
    public static final Option<Integer> GGUF_TILED_ROWS = integer("jitnn.gguf.tiledRows", 32, KERNELS, "`32`", "GGUF weights: up to this many input rows use the quantised matrix-vector kernels; more rows dequantise and use the GEMM");

    /** {@code jitnn.fuse}. */
    public static final Option<Boolean> FUSE = bool("jitnn.fuse", true, KERNELS, "`true`", "fusion before compilation: convolution + BatchNorm (+ ReLU) on folded weights, GELU in the GEMM epilogue, add + ReLU");

    /** {@code jitnn.channelsLast}. */
    public static final Option<Boolean> CHANNELS_LAST = bool("jitnn.channelsLast", true, KERNELS, "`true`", "in inference, keep image tensors around convolutions, element-wise operators and casts in [N, H, W, C] after BatchNorm folding, converting only for operators that need [N, C, H, W]");

    /** {@code jitnn.channelsLast.verbose}. */
    public static final Option<Boolean> CHANNELS_LAST_VERBOSE = bool("jitnn.channelsLast.verbose", false, KERNELS, "`false`", "log each remaining layout conversion and the operator that needs it");

    /** {@code jitnn.fuse.elementwise}. */
    public static final Option<Boolean> FUSE_ELEMENTWISE = bool("jitnn.fuse.elementwise", true, KERNELS, "`true`", "chains of FP32 unary and binary operators with internal intermediates run as one generated kernel");

    /** {@code jitnn.onnx.rewrite}. */
    public static final Option<Boolean> ONNX_REWRITE = bool("jitnn.onnx.rewrite", true, MODELS, "`true`", "`jitnn-onnx` (and `jitnn-tf`): at load, rewrite MatMul by a constant (+ bias) into linear layers, the erf GELU chain into GELU, and attention chains into one attention operator");

    /** {@code jitnn.onnx.int8}. */
    public static final Option<Boolean> ONNX_INT8 = bool("jitnn.onnx.int8", true, MODELS, "`true`", "products of quantised models with a constant INT8 weight run in INT8 (tensor cores on CUDA), dequantised in the same kernel; `false` runs them in FP32");

    /** {@code jitnn.onnx.buckets}. */
    public static final Option<String> ONNX_BUCKETS = string("jitnn.onnx.buckets", "", MODELS, "empty", "shape buckets for every ONNX session (`1,2,4,8,...`): symbolic batch and masked sequence dimensions are padded to the smallest bucket, so one plan is compiled per bucket (see `OnnxSession.withBuckets`)");

    /** {@code jitnn.onnx.halfActivations}. */
    public static final Option<Boolean> ONNX_HALF_ACTIVATIONS = bool("jitnn.onnx.halfActivations", true, MODELS, "`true`", "a half-precision session keeps activations in FP16 (`Autocast.enableHalf`); `false` keeps FP32 activations with FP16 products (`Autocast.enable`)");

    /** {@code jitnn.onnx.fusedAttentionRows}. */
    public static final Option<Integer> ONNX_FUSED_ATTENTION_ROWS = integer("jitnn.onnx.fusedAttentionRows", 32, MODELS, "`32`", "rewritten attention uses the fused kernel when batch x heads is at most this, else batched GEMMs and a fused scale-softmax");

    /** {@code jitnn.djl.fallback}. */
    public static final Option<String> DJL_FALLBACK = string("jitnn.djl.fallback", "", MODELS, "empty", "`jitnn-djl`: a DJL engine (for example `PyTorch`) on the class path that runs operators jitNN lacks; empty: those operators throw");

    /** {@code jitnn.djl.persistentBytes}. */
    public static final Option<Long> DJL_PERSISTENT_BYTES = longValue("jitnn.djl.persistentBytes", 4L << 20, MODELS, "`4194304` (4 MiB)", "`jitnn-djl`: arrays created from host data at least this large are uploaded once into persistent device storage");

    /** {@code jitnn.native}. */
    public static final Option<String> NATIVE = string("jitnn.native", null, NATIVE_LIBS, "`off`", "vendor libraries in place of jitNN kernels: `auto` (every library available on the device) or a list of `cublas`, `cudnn`, `cufft`, `cusparse`, `cutlass`, `mlx`; anything missing falls back to jitNN kernels. Under `auto`, FP32 convolutions use cuDNN only with `jitnn.native.tf32=true`; naming `cudnn` uses it for them regardless");

    /** {@code jitnn.withNative}. */
    public static final Option<Boolean> WITH_NATIVE = bool("jitnn.withNative", false, NATIVE_LIBS, "`false`", "`true` is the same as `jitnn.native=auto` (the `--with-native` option of the [jitNN-benchmarking](https://github.com/beestream-ai/jitNN-benchmarking) scripts sets it)");

    /** {@code jitnn.halfAttention}. */
    public static final Option<String> HALF_ATTENTION = choice("jitnn.halfAttention", "auto", NATIVE_LIBS, "`auto`", new String[] { "auto", "cudnn", "cutlass", "flash" }, "attention in FP16 inference (`Autocast.enableHalf`): `auto` uses cuDNN's fused attention, else CUTLASS's batched FP16 GEMMs when `jitnn.native` allows them, else the FP16 FlashAttention kernel; `cudnn`, `cutlass` or `flash` forces one");

    /** {@code jitnn.native.approximateGelu}. */
    public static final Option<Boolean> NATIVE_APPROXIMATE_GELU = bool("jitnn.native.approximateGelu", true, NATIVE_LIBS, "`true`", "with the bias epilogue, exact GELU after an FP16 linear layer runs as cuBLASLt's tanh-approximated GELU (differences up to about 5e-4, below FP16 rounding); `false` keeps exact GELU as a separate kernel");

    /** {@code jitnn.native.biasEpilogue}. */
    public static final Option<Boolean> NATIVE_BIAS_EPILOGUE = bool("jitnn.native.biasEpilogue", true, NATIVE_LIBS, "`true`", "FP16 linear layers in `Autocast.enableHalf` inference add bias (and ReLU) in cuBLASLt's GEMM epilogue; `false` runs a separate kernel");

    /** {@code jitnn.native.tf32}. */
    public static final Option<Boolean> NATIVE_TF32 = bool("jitnn.native.tf32", false, NATIVE_LIBS, "`false`", "cuBLAS FP32 products on TF32 tensor cores");

    /** {@code jitnn.native.convMinFlops}. */
    public static final Option<Long> NATIVE_CONV_MIN_FLOPS = longValue("jitnn.native.convMinFlops", 1L << 30, NATIVE_LIBS, "`1073741824` (2^30)", "fewest multiply-adds for a convolution to go to cuDNN");

    /** {@code jitnn.native.mlxConvMinFlops}. */
    public static final Option<Long> NATIVE_MLX_CONV_MIN_FLOPS = longValue("jitnn.native.mlxConvMinFlops", 1L << 28, NATIVE_LIBS, "`268435456` (2^28)", "fewest multiply-adds for a convolution to go to MLX");

    /** {@code jitnn.native.mlxCopyMinFlops}. */
    public static final Option<Long> NATIVE_MLX_COPY_MIN_FLOPS = longValue("jitnn.native.mlxCopyMinFlops", 1L << 30, NATIVE_LIBS, "`1073741824` (2^30)", "fewest multiply-adds for a product whose operands sit in larger reused buffers to go to MLX through exact-size copies");

    /** {@code jitnn.native.mlxGatherMin}. */
    public static final Option<Long> NATIVE_MLX_GATHER_MIN = longValue("jitnn.native.mlxGatherMin", 1L << 26, NATIVE_LIBS, "`67108864` (2^26)", "fewest q k^T multiply-adds for attention in another layout (packed, or [B, T, H, d]) to be gathered into [B, H, T, d] for MLX");

    /** {@code jitnn.native.cudnnChannelsLast}. */
    public static final Option<String> NATIVE_CUDNN_CHANNELS_LAST = choice("jitnn.native.cudnnChannelsLast", "all", NATIVE_LIBS, "`all`", new String[] { "all", "half", "off" }, "cuDNN convolutions in inference run channels-last (NHWC), with surrounding tensors kept channels-last: `all`, `half` (FP16 only) or `off` (NCHW)");

    /** {@code jitnn.native.attentionFp16}. */
    public static final Option<Boolean> NATIVE_ATTENTION_FP16 = bool("jitnn.native.attentionFp16", false, NATIVE_LIBS, "`false`", "FP32 attention in inference through cuDNN's fused FP16 attention (q, k, v rounded to FP16)");

    /** {@code jitnn.native.unavailable}. */
    public static final Option<String> NATIVE_UNAVAILABLE = string("jitnn.native.unavailable", "", NATIVE_LIBS, "empty", "comma-separated libraries treated as missing, to test the fallback");

    /** {@code jitnn.native.verbose}. */
    public static final Option<Boolean> NATIVE_VERBOSE = bool("jitnn.native.verbose", false, NATIVE_LIBS, "`false`", "log the libraries found on each device");

    /** Every option, in documentation order. */
    public static List<Option<?>> options() {
        return Collections.unmodifiableList(OPTIONS);
    }

    /** Explicitly set values, by option. */
    private final Map<Option<?>, Object> values;

    private JitnnConfig(Map<Option<?>, Object> values) {
        this.values = Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }

    /** The value of {@code option}: the one set, else its default (null for a machine-dependent default). */
    public <T> T get(Option<T> option) {
        Object v = values.get(option);
        return v != null ? option.type().cast(v) : option.defaultValue();
    }

    /** Whether {@code option} was set (as a system property or with the builder) rather than left at its default. */
    public boolean isSet(Option<?> option) {
        return values.containsKey(option);
    }

    /** A copy with {@code option} set to {@code value} (null: back to its default). */
    public <T> JitnnConfig with(Option<T> option, T value) {
        Map<Option<?>, Object> copy = new LinkedHashMap<>(values);
        if (value == null) {
            copy.remove(option);
        } else {
            copy.put(option, option.type().cast(value));
        }
        return new JitnnConfig(copy);
    }

    /** The options that were set, as {@code key=value} properties. */
    public Properties toProperties() {
        Properties p = new Properties();
        values.forEach((o, v) -> p.setProperty(o.key(), String.valueOf(v)));
        return p;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof JitnnConfig c && values.equals(c.values);
    }

    @Override
    public int hashCode() {
        return values.hashCode();
    }

    @Override
    public String toString() {
        return "JitnnConfig" + values;
    }

    /** All defaults. */
    public static JitnnConfig defaults() {
        return new JitnnConfig(Map.of());
    }

    /** The options set in {@code properties} ({@code jitnn.*} keys); other keys are ignored. */
    public static JitnnConfig fromProperties(Properties properties) {
        Map<Option<?>, Object> values = new LinkedHashMap<>();
        for (Option<?> o : OPTIONS) {
            String v = properties.getProperty(o.key());
            if (v != null) {
                values.put(o, o.parse(v));
            }
        }
        return new JitnnConfig(values);
    }

    /** The options set as system properties now. */
    public static JitnnConfig fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    /** A builder starting from the defaults. */
    public static Builder builder() {
        return new Builder(defaults());
    }

    /** A builder starting from the system properties. */
    public static Builder builderFromSystemProperties() {
        return new Builder(fromSystemProperties());
    }

    /** Builds a configuration option by option. */
    public static final class Builder {
        private JitnnConfig config;

        private Builder(JitnnConfig start) {
            this.config = start;
        }

        /** Sets {@code option} (null: back to its default). */
        public <T> Builder set(Option<T> option, T value) {
            config = config.with(option, value);
            return this;
        }

        /** Sets the option with this key from its text form, as a system property would. */
        public Builder set(String key, String value) {
            for (Option<?> o : OPTIONS) {
                if (o.key().equals(key)) {
                    return setParsed(o, value);
                }
            }
            throw new IllegalArgumentException("unknown jitNN option " + key);
        }

        private <T> Builder setParsed(Option<T> o, String value) {
            return set(o, value == null ? null : o.parse(value));
        }

        /** The configuration. */
        public JitnnConfig build() {
            return config;
        }
    }

    private static volatile JitnnConfig current;
    private static volatile boolean started;

    /**
     * The runtime's configuration. The first call fixes it: the {@linkplain #install installed}
     * one, else the system properties at that moment.
     */
    public static JitnnConfig current() {
        return started ? current : start();
    }

    private static synchronized JitnnConfig start() {
        if (current == null) {
            current = fromSystemProperties();
        }
        started = true;
        return current;
    }

    /**
     * Makes {@code config} the runtime's configuration. Call it before the first tensor operation;
     * once the runtime has read its configuration, installing a different one fails with an
     * {@link IllegalStateException} (use {@link #change} for the options that may change).
     */
    public static synchronized void install(JitnnConfig config) {
        Objects.requireNonNull(config);
        if (started && !config.equals(current)) {
            throw new IllegalStateException("the jitNN runtime has started with " + current + "; install a configuration before the first tensor operation");
        }
        current = config;
    }

    /**
     * Gives {@code option} a different default before the runtime starts, unless it was set:
     * for integrations such as the DJL engine. True if the value was applied.
     */
    public static synchronized <T> boolean defaultIfUnset(Option<T> option, T value) {
        if (started) {
            return false;
        }
        JitnnConfig base = current != null ? current : fromSystemProperties();
        if (base.isSet(option)) {
            current = base;
            return false;
        }
        current = base.with(option, value);
        return true;
    }

    /**
     * Changes an option that may change at run time ({@link Option#changeableAtRunTime()});
     * operations recorded afterwards see the new value. Other options fail with an
     * {@link IllegalArgumentException}.
     */
    public static synchronized <T> void change(Option<T> option, T value) {
        if (!option.changeableAtRunTime()) {
            throw new IllegalArgumentException(option.key() + " is a start-time option: set it with -D" + option.key() + " or install a configuration");
        }
        current = (current != null ? current : fromSystemProperties()).with(option, value);
    }

    /**
     * The Markdown table of a section's options (property, default, meaning), as
     * {@code docs/CONFIGURATION.md} shows it.
     */
    public static String markdownTable(Section section) {
        StringBuilder b = new StringBuilder("| Property | Default | Meaning |\n|---|---|---|\n");
        for (Option<?> o : OPTIONS) {
            if (o.section() == section) {
                b.append("| `").append(o.key()).append("` | ").append(o.defaultText()).append(" | ").append(o.description());
                if (o.changeableAtRunTime()) {
                    b.append(" (can change at run time)");
                }
                b.append(" |\n");
            }
        }
        return b.toString();
    }
}
