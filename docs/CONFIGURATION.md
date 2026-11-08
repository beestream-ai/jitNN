# Configuration

jitNN's options are `jitnn.*` Java system properties (`-D...`), or a `JitnnConfig` installed from
code before the first tensor operation:

```java
JitnnConfig.install(JitnnConfig.builder()
        .set(JitnnConfig.PLAN_CACHE, 64)
        .set(JitnnConfig.NATIVE, "auto")
        .build());
```

The runtime reads its configuration once, the first time it needs an option, and keeps it.
Options marked "can change at run time" may change afterwards with `JitnnConfig.change` or their
setters (`JitnnRuntime.setDeterministic`, `JitnnRuntime.setAutotune`); the others are fixed from then on. An
invalid value (a number that does not parse, a value outside an option's choices) fails with an
error naming the property. Sizes are in bytes.

The tables below are generated from `JitnnConfig` (`ConfigurationDocTest` checks them; run it
with `-Djitnn.docs.write=true` to rewrite them).

## Runtime options

<!-- generated: RUNTIME -->
| Property | Default | Meaning |
|---|---|---|
| `jitnn.backend` | `tornado` with a TornadoVM runtime, else `cpu` with `--add-modules jdk.incubator.vector`, else `reference` | where operators run: `tornado` (a TornadoVM device), `cpu` (the host, multithreaded, Java Vector API; operators without a CPU kernel run their reference) or `reference` (the host, scalar, one operator at a time) |
| `jitnn.device` | TornadoVM's default device | `<driver>:<device>`, as listed by `tornado --devices` |
| `jitnn.concurrency` | `1` | concurrent execution slots, each with its own plans and device buffers (weights included); threads are spread over them. Plans on different devices always run concurrently. `jitnn.plan.cache` applies per slot |
| `jitnn.cpu.threads` | the physical cores this process may use, at most one socket's; on Apple silicon the performance cores | threads of the `cpu` backend's parallel loops, the calling thread included |
| `jitnn.cpu.spinMicros` | `200` (`0` on macOS) | how long idle `cpu` backend threads (and a waiting caller) spin before sleeping |
| `jitnn.cpu.poolBytes` | `2147483648` (2 GiB) | host arrays the `cpu` backend keeps for reuse, unless `jitnn.hostPool.bytes` is set |
| `jitnn.cpu.gemm.kc` | `256` | `cpu` GEMM blocking: depth of A and B per block |
| `jitnn.cpu.gemm.mcPanels` | `16` | `cpu` GEMM blocking: rows of A per tile, in panels of 6 |
| `jitnn.cpu.gemm.nc` | `256` | `cpu` GEMM blocking: columns of B per tile |
| `jitnn.cpu.gemm.tileMacs` | `1048576` | `cpu` GEMM: fewest multiply-adds per parallel tile |
| `jitnn.cpu.opencl` | `auto` | where the `cpu` backend runs operators it has no kernel for. `auto`: the first OpenCL CPU device (PoCL, Intel CPU runtime) when a TornadoVM runtime is present and the JVM runs with `LD_PRELOAD=$JAVA_HOME/lib/libjsig.so` (without it the drivers' signal handlers crash the JVM). Give the driver the backend's thread count: `POCL_MAX_PTHREAD_COUNT`, or `POCL_CPU_MAX_CU_COUNT` after PoCL 3; `CL_CONFIG_CPU_TBB_NUM_WORKERS` for Intel. `off`: the reference. `<driver>:<device>`: that device. Each operator and shape is timed on the reference and the device and keeps the faster; consecutive device operators run as one plan; one the device fails to compile (FP16 without `cl_khr_fp16`) stays on the reference |
| `jitnn.cpu.opencl.choose` | `measure` | `always`: send operators without a CPU kernel to the `jitnn.cpu.opencl` device without timing (tests) (can change at run time) |
| `jitnn.cpu.blas` | `auto` | native BLAS (`cblas_sgemm`) for the `cpu` backend's linear, matmul and convolutions, forward and backward: `auto` tries MKL, OpenBLAS, Arm Performance Libraries and Accelerate on the library path; `off`; or a library file. Chosen per operator and shape by `jitnn.cpu.native.choose` |
| `jitnn.cpu.onednn` | `auto` | oneDNN (`libdnnl`) for the `cpu` backend's convolutions (forward with bias and ReLU, both gradients, grouped): `auto`, `off` or a library file. Use an OpenMP build, whose thread count jitNN sets; a TBB build takes every core |
| `jitnn.cpu.bnns` | `auto` | on macOS, BNNS (Accelerate) for the same convolutions; `off` never |
| `jitnn.cpu.native.choose` | `measure` | how the `cpu` backend picks between its kernel and `blas`, `onednn`, `bnns` per operator and shape: `measure` times each way for three calls, runs the fastest, and re-times after 64 more calls for the final choice; `kernel`, `blas`, `onednn` or `bnns` forces that way when available (tests) (can change at run time) |
| `jitnn.cpu.reuse` | `true` | a `cpu` trace of the same structure as the last reuses its intermediates' arrays, as a compiled plan does; tensors not requested at that sync (`JitnnRuntime.sync(t)`, `t.item()`) are then gone |
| `jitnn.cpu.reuse.traces` | `32` | trace structures each thread remembers for that reuse |
| `jitnn.cpu.stats` | `false` | time every `cpu` operator and print calls, time and reference fallbacks at exit; `JitnnRuntime.cpuStats()` gives counts without it |
| `jitnn.cpu.stats.shapes` | `false` | with `jitnn.cpu.stats`: one row per operator and shape |
| `jitnn.plan.cache` | `16` (`64` under `jitnn-djl` when not set) | compiled plans kept; the least recently used is closed, saving parameters first |
| `jitnn.plan.idleBytes` | `67108864` (64 MiB) | a plan holding at least this much intermediate memory is closed after `jitnn.plan.idleExecutions` executions without it, and recompiled if needed again |
| `jitnn.plan.idleExecutions` | `256` | executions of other plans before an idle large plan is closed (checked every 32) |
| `jitnn.plan.carry` | `true` | an input that is an output of the same plan's previous execution (a gradient accumulated without `zeroGrad`) is copied on the device instead of round-tripping through the host; the plan recompiles once when this is first seen |
| `jitnn.plan.memo` | `true` | a trace recorded again with the same operations, shapes and weights replays its last plan without rewriting the trace or building its key, once that plan has run twice; not for traces with a large host input |
| `jitnn.promoteInputBytes` | `16777216` (16 MiB) | a host input at least this large that a later execution reads again becomes persistent device storage |
| `jitnn.cudaGraph` | `true` | on CUDA, capture each plan as a CUDA graph and replay it |
| `jitnn.dynamicShapes` | `true` | share kernels across array sizes (TornadoVM `withDynamicShapes()`), so a new batch or sequence size reuses compiled kernels |
| `jitnn.graph.maxTasks` | `40` | tasks per TornadoVM task graph; longer traces become a chain of graphs (a graph holds at most 256 tasks and 8 KiB of bytecode) |
| `jitnn.hostPool.bytes` | `268435456` (256 MiB) | host arrays kept for reuse: outputs that outlive their plan's next run are copied into one, returned on `Tensor.release` or collection |
| `jitnn.parallelCopyBytes` | `33554432` (32 MiB) | host inputs at least this large are copied into a plan's buffers by up to eight threads |
| `jitnn.reuseBuffers` | `true` | reuse dead backward-pass and inference buffers inside a compiled step |
| `jitnn.reuseBuffers.exact` | `auto` | reuse only dead buffers of exactly the same size, so vendor-library operands need no copies: `auto` (on Metal with MLX), `true`, `false` |
| `jitnn.verbose` | `false` | log every compiled plan (device, nodes, tasks, build time) and the vendor libraries found |
| `jitnn.debugKeys` | `false` | print each new plan's cache key to standard error, to find what makes a trace recompile |
| `jitnn.printKernels` | `false` | print each compiled plan's operators and kernel source: `true` for standard output, or a file name (appended); see [Printing the kernels](USAGE.md#printing-the-kernels) |
| `jitnn.profile` | `false` | TornadoVM's profiler: `JitnnRuntime.lastProfile()` gives the last step's kernel time, `JitnnRuntime.kernelBreakdown()` device time per operator task. `JitnnRuntime.startTrace`/`stopTrace` write a Chrome trace of plan runs and kernels. On CUDA also set `jitnn.cudaGraph=false`: the profiler reports no kernel time under CUDA graphs |
<!-- end generated -->

`JitnnRuntime.planStats()` (plans compiled, replayed, reused) and `JitnnRuntime.transferStats()` (fetches,
re-uploads, input copies) need no option.

## Kernel variants

The defaults suit most workloads; the other values are for comparison and for working around a
backend problem.

<!-- generated: KERNELS -->
| Property | Default | Meaning |
|---|---|---|
| `jitnn.gemm` | `auto` | FP32 GEMM kernel (see [GEMM selection](#gemm-selection-auto)): `auto`, `simdgroup`, `128`, `64`, `16`, or padded `gemm64p`, `gemm64pd`, `gemm128p`, `gemm128pd`, `gemm128x64pd`, `gemm64x128pd`. A forced variant also disables forward split-K (can change at run time) |
| `jitnn.tuning` | empty | GEMM tuning table: empty reads `tuning/<backend>-<device>.json` from the class path (tables ship for the A10 and A100 80GB PCIe), a path reads that file, `off` ignores tables. Used by `jitnn.gemm=auto` on CUDA and OpenCL |
| `jitnn.fuse.program` | `true` | fused element-wise chains of up to six operations sharing the output's layout use a kernel specialised for the chain (`FusedKernels.fusedProgram1` to `6`); `false` uses the generic interpreter kernel |
| `jitnn.deterministic` | `false` | reproducible runs: no autotuning measurements, deterministic cuDNN algorithms (`JitnnRuntime.setDeterministic`) (can change at run time) |
| `jitnn.autotune` | `off` | run-time autotuning: `on` measures the variants of each new GEMM, 1x1 and 3x3 convolution and row-norm shape on first use and keeps the fastest; `use` only applies saved decisions. Decisions override tuning tables and size rules (can change at run time) |
| `jitnn.autotune.dir` | `~/.jitnn/autotune` | where decisions are saved, one `<backend>-<device>.properties` per device (can change at run time) |
| `jitnn.autotune.margin` | `0.05` | how much faster a variant must be to replace the rule's choice |
| `jitnn.autotune.reps` | `8` | operator copies per autotuning measurement plan |
| `jitnn.autotune.runs` | `15` | timed executions of a measurement plan (the median is used) |
| `jitnn.autotune.verbose` | `false` | print each measurement and decision |
| `jitnn.splitK` | `true` | split the reduction of GEMMs and convolutions whose output fills too few work-groups (A^T B weight gradients, forward products), summing partials in order |
| `jitnn.splitK.groups` | `256` | work-groups a forward product should reach before it stops splitting K |
| `jitnn.tensorCores` | `true` | on CUDA, FP16 and BF16 products run on tensor cores (`mma.sync` m16n8k16, compute capability 8.0+) |
| `jitnn.transpose` | `32` | transpose kernel: `32` (32x32 tiles), `16` or `naive` (can change at run time) |
| `jitnn.binaryRepeat` | `true` | a binary operator whose broadcast operand repeats one contiguous block uses an index kernel (`(i / div) % mod`) instead of the general broadcast kernel |
| `jitnn.fusedAttention` | `true` | FP32 attention with head dimension up to 128 uses the fused FlashAttention-style kernels; `false` composes batched GEMMs and a softmax |
| `jitnn.flash` | tiled | `simple`: head dimensions up to 64 use the generic fused attention kernels instead of the tiled ones |
| `jitnn.flash.mma` | `true` | on CUDA with `jitnn.tensorCores`, FP16 attention (forward and backward, head dimension up to 128) runs on tensor cores; `false` uses the tiled FP16 kernels up to 64 and FP32 copies above |
| `jitnn.rownorm` | warp | `group`: LayerNorm and RMSNorm on CUDA use one work-group per row instead of one warp (used from 1024 rows of at most 768 features) |
| `jitnn.conv` | padded | `simple`: plain implicit-GEMM convolution kernels instead of the padded, prefetching ones |
| `jitnn.winograd` | `auto` | 3x3 stride-1 convolutions (one group, at least 16 channels in and out, padding up to 2) use Winograd F(4x4, 3x3), or F(2x2, 3x3) with too few 4x4 tiles, forward and both gradients. `2` keeps F(2x2, 3x3), `false` disables Winograd |
| `jitnn.conv1x1` | GEMM | `implicit`: 1x1 stride-1 convolutions use the implicit-GEMM kernels instead of a batched GEMM over the images |
| `jitnn.conv1x1.gemm` | `auto` | 1x1 convolution GEMM tile: `64`, `128x64`, `64x128`, `128x16` (for 4x4 planes), or `auto` (by output rows and plane size) |
| `jitnn.conv1x1.backward` | `gemm` | `implicit`: the 1x1 input gradient uses the implicit-GEMM kernel instead of the batched GEMM |
| `jitnn.convPhases` | `true` | input gradient of strided convolutions as a GEMM and a gather (when the column buffer fits in 256 MB), else one implicit GEMM per stride phase; `false` uses the general gathering kernel |
| `jitnn.conv.weightGroups` | `256` | the filter gradient is split over output positions until it reaches about this many work-groups |
| `jitnn.conv.weightChunk` | `256` | fewest output positions per slice of that split |
| `jitnn.rnn` | `fused` | LSTM layers of up to 512 hidden units and 256 batch rows run the whole sequence in one launch (forward and backward); `steps` launches tiled kernels per timestep, `simple` thread-per-cell kernels. GRU always runs per timestep (tiled, or thread-per-cell with `simple`) |
| `jitnn.gguf.tiledRows` | `32` | GGUF weights: up to this many input rows use the quantised matrix-vector kernels; more rows dequantise and use the GEMM |
| `jitnn.fuse` | `true` | fusion before compilation: convolution + BatchNorm (+ ReLU) on folded weights, GELU in the GEMM epilogue, add + ReLU |
| `jitnn.channelsLast` | `true` | in inference, keep image tensors around convolutions, element-wise operators and casts in [N, H, W, C] after BatchNorm folding, converting only for operators that need [N, C, H, W] |
| `jitnn.channelsLast.verbose` | `false` | log each remaining layout conversion and the operator that needs it |
| `jitnn.fuse.elementwise` | `true` | chains of FP32 unary and binary operators with internal intermediates run as one generated kernel |
<!-- end generated -->

## Imported models and DJL

<!-- generated: MODELS -->
| Property | Default | Meaning |
|---|---|---|
| `jitnn.onnx.rewrite` | `true` | `jitnn-onnx` (and `jitnn-tf`): at load, rewrite MatMul by a constant (+ bias) into linear layers, the erf GELU chain into GELU, and attention chains into one attention operator |
| `jitnn.onnx.int8` | `true` | products of quantised models with a constant INT8 weight run in INT8 (tensor cores on CUDA), dequantised in the same kernel; `false` runs them in FP32 |
| `jitnn.onnx.buckets` | empty | shape buckets for every ONNX session (`1,2,4,8,...`): symbolic batch and masked sequence dimensions are padded to the smallest bucket, so one plan is compiled per bucket (see `OnnxSession.withBuckets`) |
| `jitnn.onnx.halfActivations` | `true` | a half-precision session keeps activations in FP16 (`Autocast.enableHalf`); `false` keeps FP32 activations with FP16 products (`Autocast.enable`) |
| `jitnn.onnx.fusedAttentionRows` | `32` | rewritten attention uses the fused kernel when batch x heads is at most this, else batched GEMMs and a fused scale-softmax |
| `jitnn.djl.fallback` | empty | `jitnn-djl`: a DJL engine (for example `PyTorch`) on the class path that runs operators jitNN lacks; empty: those operators throw |
| `jitnn.djl.persistentBytes` | `4194304` (4 MiB) | `jitnn-djl`: arrays created from host data at least this large are uploaded once into persistent device storage |
<!-- end generated -->

## Vendor libraries

See [NATIVE.md](NATIVE.md) for what each library takes and how jitNN falls back to its kernels.

<!-- generated: NATIVE -->
| Property | Default | Meaning |
|---|---|---|
| `jitnn.native` | `off` | vendor libraries in place of jitNN kernels: `auto` (every library available on the device) or a list of `cublas`, `cudnn`, `cufft`, `cusparse`, `cutlass`, `mlx`; anything missing falls back to jitNN kernels. Under `auto`, FP32 convolutions use cuDNN only with `jitnn.native.tf32=true`; naming `cudnn` uses it for them regardless |
| `jitnn.withNative` | `false` | `true` is the same as `jitnn.native=auto` (the `--with-native` option of the [jitNN-benchmarking](https://github.com/beestream-ai/jitNN-benchmarking) scripts sets it) |
| `jitnn.halfAttention` | `auto` | attention in FP16 inference (`Autocast.enableHalf`): `auto` uses cuDNN's fused attention, else CUTLASS's batched FP16 GEMMs when `jitnn.native` allows them, else the FP16 FlashAttention kernel; `cudnn`, `cutlass` or `flash` forces one |
| `jitnn.native.approximateGelu` | `true` | with the bias epilogue, exact GELU after an FP16 linear layer runs as cuBLASLt's tanh-approximated GELU (differences up to about 5e-4, below FP16 rounding); `false` keeps exact GELU as a separate kernel |
| `jitnn.native.biasEpilogue` | `true` | FP16 linear layers in `Autocast.enableHalf` inference add bias (and ReLU) in cuBLASLt's GEMM epilogue; `false` runs a separate kernel |
| `jitnn.native.tf32` | `false` | cuBLAS FP32 products on TF32 tensor cores |
| `jitnn.native.convMinFlops` | `1073741824` (2^30) | fewest multiply-adds for a convolution to go to cuDNN |
| `jitnn.native.mlxConvMinFlops` | `268435456` (2^28) | fewest multiply-adds for a convolution to go to MLX |
| `jitnn.native.mlxCopyMinFlops` | `1073741824` (2^30) | fewest multiply-adds for a product whose operands sit in larger reused buffers to go to MLX through exact-size copies |
| `jitnn.native.mlxGatherMin` | `67108864` (2^26) | fewest q k^T multiply-adds for attention in another layout (packed, or [B, T, H, d]) to be gathered into [B, H, T, d] for MLX |
| `jitnn.native.cudnnChannelsLast` | `all` | cuDNN convolutions in inference run channels-last (NHWC), with surrounding tensors kept channels-last: `all`, `half` (FP16 only) or `off` (NCHW) |
| `jitnn.native.attentionFp16` | `false` | FP32 attention in inference through cuDNN's fused FP16 attention (q, k, v rounded to FP16) |
| `jitnn.native.unavailable` | empty | comma-separated libraries treated as missing, to test the fallback |
| `jitnn.native.verbose` | `false` | log the libraries found on each device |
<!-- end generated -->

## TornadoVM properties jitNN sets

jitNN sets these TornadoVM system properties itself; set one explicitly to keep another value.

| Property | Set to | Why |
|---|---|---|
| `tornado.cuda.payloadAlignment` | `128` when `jitnn.native` allows a CUDA library, else TornadoVM's `32` | TornadoVM's alignment of array data on CUDA devices, in bytes. cuBLAS picks different tensor-core kernels for 32-byte-aligned operands |
| `tornado.mlx.async` | `true` when `jitnn.native` allows MLX | MLX calls commit without waiting for the GPU; TornadoVM still waits before host reads |
| `tornado.cudnn.deterministic` | the value of `jitnn.deterministic` | cuDNN picks only deterministic algorithms |
| `<task graph>.device` | the thread's device | TornadoVM builds a task's code for the graph's device when the task is added, before `withDevice` on the plan applies |

## GEMM selection (`auto`)

| Condition | Kernel |
|---|---|
| a vendor library is enabled and takes the product | cuBLAS or MLX (see [NATIVE.md](NATIVE.md)) |
| Metal, M and N multiples of 8, output at least 64x64 and K at least 16 | `gemmSimdgroup`: simdgroup 8x8 matrix units, 64x64 per 128 threads; the bias/ReLU epilogue is a second kernel |
| output at least 64x64 but too few 64x64 tiles to fill the GPU | `gemm64SplitK`: K split into slices (up to `jitnn.splitK.groups` work-groups, at least 64 of K each), summed in order with the epilogue |
| the device's tuning table has an entry for the shape, or for its class (ceiling log2 of M, N and K, and the transposes) | that variant |
| output at least 64x64 and K at least 16 | `gemm64pd`: 64x64 per 256 threads, padded local memory, next tile prefetched into registers |
| otherwise | `gemm`: 16x16 tiles, one output per thread |

All variants take both transposes and an epilogue (bias, ReLU) without materialising anything.
Half-precision operands (FP16, BF16) run on tensor cores on CUDA (a 128x128 tile when M and N are
multiples of 128 and K of 16), on simdgroup kernels on Metal, and on 64x64 register-blocked
kernels elsewhere, converting on load. A^T B products with a long reduction and a small output
(weight gradients) are split along K into up to 32 slices. The tuning tables are written by
`scripts/tune.py` in jitNN-benchmarking.

## Test properties

Maven properties of the `accel-tests` profile, and system properties read by the tests (pass the
latter through `jitnn.test.extra`).

| Property | Default | Meaning |
|---|---|---|
| `jitnn.test.device` | `0:0` | device for the accelerator tests |
| `jitnn.test.backend` | `tornado` | backend `OpsAccelTest` and `GoldenAccelTest` check against the reference: `tornado` or `cpu` |
| `jitnn.test.verbose` | `false` | passes `jitnn.verbose` |
| `jitnn.test.profile` | `false` | passes `jitnn.profile` |
| `jitnn.test.gemm` | `auto` | passes `jitnn.gemm` |
| `jitnn.test.extra` | empty | extra JVM arguments, e.g. `-Dtornado.printKernel=True` or `-Djitnn.native=auto` |
| `jitnn.test.deviceMemory` | `12GB` | TornadoVM's device-memory budget (`tornado.device.memory`, whose own default is 4GB) |
| `jitnn.test.keepPlans` | `false` | `CnnAccelTest` keeps compiled plans between tests |
| `jitnn.mnist` | empty | MNIST directory; the MNIST gate is skipped when empty |
| `jitnn.cifar10` | empty | directory holding `cifar-10-batches-bin`; the CIFAR-10 gates are skipped when empty |
| `jitnn.text` | empty | directory holding `corpus.txt` and the transformer references; the transformer gates are skipped when empty |
| `jitnn.onnx.exportDir` | empty | `OnnxExportTest` and `OnnxExportAccelTest` keep the exported files, inputs and outputs there for `scripts/onnx/check_export.py` |
| `jitnn.hf` | empty | directory written by `scripts/hf/hf_reference.py`; `HfAccelTest` is skipped when empty |
| `jitnn.transformer.steps` | 300 | training steps of the transformer gates (must match the PyTorch reference) |
| `jitnn.dist.devices` | empty | comma-separated devices of the collective and data-parallel tests, one rank each, e.g. `0:0,1:0`; empty: two ranks on the test device |
| `jitnn.dist.nccl` | `false` | `DataParallelAccelTest`: also run on `NcclCollective` (distinct CUDA devices) |
| `jitnn.bench.dist` | `false` | `DataParallelBenchAccelTest`: training throughput on one device and on `jitnn.dist.devices` with the host collective and NCCL; `jitnn.bench.dist.model` (`resnet18`, `resnet50`), `.batch`, `.steps`, `.modes` |
| `jitnn.bench.dim`, `.layers`, `.seq`, `.batch` | 128, 4, 128, 16 | model and batch sizes of `TransformerBenchAccelTest` |
| `jitnn.bench.shapes`, `jitnn.bench.tasks` | `false`, empty | with `jitnn.test.profile`, list device time per GEMM shape, or per task of operators whose name contains the text |
| `jitnn.bench.inference` | `false` | run `InferenceBenchAccelTest` (ResNet-50 and BERT-base latency) |
| `jitnn.bench.models` | `resnet50,bert` | models of `InferenceBenchAccelTest` |
| `jitnn.bench.only` | empty | `InferenceBenchAccelTest`: only configurations whose name contains this text (`bert_b32_s128_fp16h`) |
| `jitnn.bench.threads` | `1` | `InferenceBenchAccelTest`: also run this many threads at once and print total samples per second (with `jitnn.concurrency` slots) |
| `jitnn.bench.split` | `false` | `InferenceBenchAccelTest`: also print the split of a call's time (recording, execution, fetching, conversion) |
| `jitnn.bench.profileTasks` | `false` | `InferenceBenchAccelTest` with `jitnn.profile`: also print device time per task |
| `jitnn.bench.warmup`, `jitnn.bench.runs` | 10, 50 | warm-up and timed calls of `InferenceBenchAccelTest` |
| `jitnn.bench.out` | empty | directory where `InferenceBenchAccelTest` writes its outputs, for `scripts/inference.py` in jitNN-benchmarking to compare |
| `jitnn.onnx.inference` | empty | directory with the exported ResNet-50 and BERT-base ONNX files and ONNX Runtime's outputs; `OnnxInferenceAccelTest` is skipped when empty |
| `jitnn.onnx.inference.match` | empty | run only the ONNX files whose name contains this text |
| `jitnn.probe` | `false` | run `GradProbeTest`: MobileNetV2 step-0 gradients on the reference backend against PyTorch FP64, per parameter (needs `jitnn.mnist`) |

`NativeAccelTest` runs only when vendor libraries are requested
(`-Djitnn.test.extra=-Djitnn.native=auto`).

Applications use TornadoVM's device-memory budget too: pass `-Dtornado.device.memory=<N>GB` for
large models. When a plan runs out of it, jitNN closes the other cached plans and recompiles.
