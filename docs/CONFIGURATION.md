# Configuration

All options are Java system properties (`-D...`), read once when the class that uses them loads.
Sizes are in bytes.

## Runtime options

| Property | Default | Meaning |
|---|---|---|
| `jitnn.backend` | `tornado` when a TornadoVM runtime is present, else `reference` | where operators run: `tornado` or `reference` |
| `jitnn.device` | TornadoVM's default device | `<driver>:<device>`, as listed by `tornado --devices` |
| `jitnn.plan.cache` | `16` (`64` under `jitnn-djl` when not set) | compiled plans kept; the least recently used is closed, saving parameters first |
| `jitnn.plan.idleBytes` | `67108864` (64 MiB) | a plan holding at least this much intermediate memory is closed once `jitnn.plan.idleExecutions` executions of other plans pass without it (a first step that never repeats, an evaluation pass between epochs); it is compiled again if it comes back |
| `jitnn.plan.idleExecutions` | `256` | executions of other plans before an idle large plan is closed (checked every 32 executions) |
| `jitnn.promoteInputBytes` | `16777216` (16 MiB) | a host input at least this large that a later execution reads again (a dataset, a constant table) becomes persistent device storage instead of being copied in on every execution |
| `jitnn.concurrency` | `1` | concurrent executions: threads are spread over this many slots, each with its own compiled plans and device buffers (weights included), so up to this many threads run plans on the device at once; plans on different devices (data-parallel ranks) always run at the same time. The plan cache holds `jitnn.plan.cache` plans per slot |
| `jitnn.cudaGraph` | `true` | on CUDA, capture each plan as a CUDA graph and replay it |
| `jitnn.graph.maxTasks` | `40` | tasks per TornadoVM task graph; longer traces become a chain of graphs (a graph holds at most 256 tasks and 8 KiB of bytecode) |
| `jitnn.reuseBuffers` | `true` | reuse dead backward-pass and inference buffers inside a compiled step |
| `jitnn.verbose` | `false` | log every compiled plan (device, nodes, tasks, build time) and the vendor libraries found |
| `jitnn.debugKeys` | `false` | print the cache key of every newly compiled plan to standard error, to find what makes a trace recompile |
| `jitnn.printKernels` | `false` | print each compiled plan's operators and generated kernel source: `true` for standard output, or a file name (appended); see [Printing the kernels](USAGE.md#printing-the-kernels) |
| `jitnn.profile` | `false` | enable TornadoVM's profiler: `Engine.lastProfile()` gives the last step's kernel time, `Engine.kernelBreakdown()` the device time per operator task (GEMMs and convolutions labelled with their shapes); on CUDA also set `jitnn.cudaGraph=false`, as the profiler reports no kernel time under CUDA graphs |

`Engine.planStats()` counts the plans compiled, replayed and reused, and `Engine.transferStats()`
the host-device traffic (fetches, re-uploads of persistent storage, input copies); neither needs an
option.

## Kernel variants

The defaults are the fastest variants measured; the other values are for comparison and for
working around a backend problem.

| Property | Default | Meaning |
|---|---|---|
| `jitnn.gemm` | `auto` | FP32 GEMM kernel (see below): `auto`, `simdgroup`, `128`, `64`, `16`, or a padded variant `gemm64p`, `gemm64pd`, `gemm128p`, `gemm128pd`, `gemm128x64pd`, `gemm64x128pd`. A forced variant also turns off forward split-K |
| `jitnn.tuning` | empty | GEMM tuning table: empty reads `tuning/<backend>-<device>.json` from the class path (jitNN ships tables for the A10 and the A100 80GB PCIe), a path reads that file, `off` ignores tables. Used by `jitnn.gemm=auto` on CUDA and OpenCL |
| `jitnn.fuse.program` | `true` | fused element-wise chains of at most six operations whose inputs share the output's layout compile to a kernel specialised for the chain (`FusedKernels.fusedProgram1` to `6`); `false` runs them on the generic interpreter kernel (faster to compile, slower to run) |
| `jitnn.autotune` | `off` | run-time autotuning (`engine.Autotune`): `on` measures the kernel variants of each new GEMM, 1x1 and 3x3 convolution and row-norm shape on the device the first time it runs and keeps the fastest; `use` only applies decisions saved earlier. Decisions override `jitnn.tuning` tables and the size rules |
| `jitnn.autotune.dir` | `~/.jitnn/autotune` | where decisions are saved: one `<backend>-<device>.properties` file per device, read by later runs |
| `jitnn.autotune.margin` | `0.05` | a measured variant replaces the rule's only when it is this much faster |
| `jitnn.autotune.reps`, `jitnn.autotune.runs` | `8`, `15` | operator copies per measurement plan, and timed executions of it (median) |
| `jitnn.autotune.verbose` | `false` | print each measurement and decision |
| `jitnn.splitK` | `true` | split the reduction of GEMMs and convolutions whose output fills too few work-groups (A^T B weight gradients, and forward products), summing the partials in order |
| `jitnn.splitK.groups` | `256` | work-groups a forward product should reach before it stops splitting K (about two per SM on an A100) |
| `jitnn.tensorCores` | `true` | on CUDA, half-precision products (FP16, BF16 operands) run on tensor cores (`mma.sync` m16n8k16, compute capability 8.0+) |
| `jitnn.transpose` | `32` | transpose kernel: `32` (32x32 tiles), `16`, or `naive` |
| `jitnn.binaryRepeat` | `true` | a binary operator whose broadcast operand repeats one contiguous block of the output's dimensions uses an index kernel (`(i / div) % mod`) instead of the general broadcast kernel; `false` turns it off |
| `jitnn.fusedAttention` | `true` | FP32 attention with head dimensions up to 128 runs the fused FlashAttention-style kernels; `false` composes batched GEMMs and a softmax |
| `jitnn.flash` | tiled | `simple`: head dimensions 33 to 64 use the generic fused attention kernels instead of the tiled ones |
| `jitnn.rownorm` | warp | `group`: LayerNorm and RMSNorm on CUDA use one work-group per row instead of one warp per row (used from 1024 rows of at most 768 features) |
| `jitnn.conv` | padded | `simple`: convolutions use the plain implicit-GEMM kernels instead of the padded, prefetching ones |
| `jitnn.winograd` | `auto` | 3x3 stride-1 convolutions (one group, at least 16 channels in and out, padding up to 2) run as Winograd F(4x4, 3x3), or F(2x2, 3x3) where there are too few 4x4 tiles, in the forward pass and both gradients. `2` keeps F(2x2, 3x3) everywhere, `false` turns Winograd off |
| `jitnn.conv1x1` | GEMM | `implicit`: 1x1 stride-1 convolutions use the implicit-GEMM kernels instead of a batched GEMM over the images |
| `jitnn.conv1x1.gemm` | `auto` | tile of the 1x1 convolution GEMM: `64`, `128x64`, `64x128`, `128x16` (narrow, for 4x4 planes), or `auto` (by output rows and plane size) |
| `jitnn.conv1x1.backward` | `gemm` | `implicit`: the input gradient of 1x1 convolutions uses the implicit-GEMM kernel instead of the batched GEMM |
| `jitnn.convPhases` | `true` | the input gradient of strided convolutions as a GEMM and a gather (when its column buffer fits in 256 MB), else one implicit GEMM per stride phase; `false` uses the general gathering kernel |
| `jitnn.conv.weightGroups` | `256` | the filter gradient is split over output positions until it reaches about this many work-groups |
| `jitnn.conv.weightChunk` | `256` | fewest output positions per slice of the filter-gradient split |
| `jitnn.rnn` | `fused` | LSTM layers of up to 512 hidden units and 256 batch rows run the whole sequence in one launch (forward and backward); `steps` launches the tiled kernels once per timestep, `simple` the thread-per-cell kernels. GRU layers always run per timestep (tiled, or thread-per-cell with `simple`) |
| `jitnn.gguf.tiledRows` | `32` | GGUF weights: up to this many input rows use the quantised matrix-vector kernels; more rows dequantise the weight and use the GEMM |
| `jitnn.fuse` | `true` | operator fusion before compilation: convolution + BatchNorm (+ ReLU) on folded weights, GELU in the GEMM epilogue, add + ReLU in one kernel |
| `jitnn.fuse.elementwise` | `true` | chains of FP32 unary and binary operators whose intermediates nothing else reads run as one generated kernel |

## Imported models and DJL

| Property | Default | Meaning |
|---|---|---|
| `jitnn.onnx.rewrite` | `true` | `jitnn-onnx` (and `jitnn-tf`, which goes through it): rewrite MatMul by a constant (+ bias) into linear layers, the erf GELU chain into GELU and attention chains into one attention operator at load |
| `jitnn.onnx.buckets` | empty | shape buckets for every ONNX session (`1,2,4,8,...`): symbolic batch and masked sequence dimensions are padded to the smallest bucket, so a server compiles one plan per bucket (see `OnnxSession.withBuckets`) |
| `jitnn.onnx.fusedAttentionRows` | `32` | `jitnn-onnx`: rewritten attention uses the fused kernel when batch x heads is at most this, else batched GEMMs and a fused scale-softmax |
| `jitnn.djl.fallback` | empty | `jitnn-djl`: a DJL engine (for example `PyTorch`) that runs the operators jitNN lacks, when it is on the class path; empty: those operators throw |
| `jitnn.djl.persistentBytes` | `4194304` (4 MiB) | `jitnn-djl`: arrays created from host data at least this large (datasets) are kept in persistent device storage, uploaded once instead of by every step that reads them |

## Vendor libraries

See [NATIVE.md](NATIVE.md) for what each library takes and the fallback to jitNN's kernels.

| Property | Default | Meaning |
|---|---|---|
| `jitnn.native` | `off` | vendor libraries in place of jitNN kernels: `auto` uses every one available on the device, or a list of `cublas`, `cudnn`, `cufft`, `cusparse`, `cutlass`, `mlx`; anything missing falls back to jitNN kernels |
| `jitnn.halfAttention` | `auto` | attention in FP16 inference (`Autocast.enableHalf`): `auto` uses cuDNN's fused attention, else CUTLASS's batched FP16 GEMMs, when `jitnn.native` makes them available, else the FP16 FlashAttention kernel; `cudnn`, `cutlass` or `flash` forces one |
| `jitnn.withNative` | `false` | `true` is `jitnn.native=auto`; the `--with-native` option of jitNN's tools sets it |
| `jitnn.native.approximateGelu` | `true` | with the bias epilogue, exact GELU after an FP16 linear layer runs as cuBLASLt's tanh-approximated GELU (differences of at most about 5e-4, below FP16's rounding); `false` keeps the exact GELU as a separate kernel |
| `jitnn.native.biasEpilogue` | `true` | FP16 linear layers in `Autocast.enableHalf` inference add their bias (and ReLU) in cuBLASLt's GEMM epilogue; `false` runs the bias as a separate kernel |
| `jitnn.native.tf32` | `false` | cuBLAS FP32 products on TF32 tensor cores |
| `jitnn.native.convMinFlops` | `1073741824` (2^30) | fewest multiply-adds for a convolution to go to cuDNN; smaller ones keep jitNN's kernels |
| `jitnn.native.cudnnChannelsLast` | `all` | cuDNN convolutions in inference run channels-last (NHWC), with the tensors around them kept channels-last: `all`, `half` (FP16 convolutions only) or `off` (NCHW) |
| `jitnn.native.attentionFp16` | `false` | FP32 attention in inference through cuDNN's fused FP16 attention (q, k and v rounded to FP16) |
| `jitnn.native.unavailable` | empty | comma-separated libraries treated as missing, to test the fallback |
| `jitnn.native.verbose` | `false` | log which libraries were found on each device |

## GEMM selection (`auto`)

| Condition | Kernel |
|---|---|
| a vendor library is enabled and takes the product | cuBLAS or MLX (see [NATIVE.md](NATIVE.md)) |
| Metal, M and N multiples of 8, output at least 64x64 and K at least 16 | `gemmSimdgroup`: simdgroup 8x8 matrix units, 64x64 per 128 threads; the bias/ReLU epilogue runs as a second kernel |
| output at least 64x64 but too few 64x64 tiles to fill the GPU | `gemm64SplitK`: K split into slices (up to `jitnn.splitK.groups` work-groups, at least 64 of K each), summed in order with the epilogue |
| the device's tuning table has an entry for the shape, or for its class (ceiling log2 of M, N and K, and the transposes) | that variant |
| output at least 64x64 and K at least 16 | `gemm64pd`: 64x64 per 256 threads, padded local memory, the next tile prefetched into registers |
| otherwise | `gemm`: 16x16 tiles, one output per thread |

All variants take both transposes and an epilogue (bias, ReLU) without materialising anything.
Half-precision operands (FP16, BF16) run on tensor cores on CUDA (a 128x128 tile when M and N are
multiples of 128 and K of 16), on the simdgroup kernels on Metal, and on 64x64 register-blocked
kernels elsewhere, converting on load. A^T B products with a long reduction and a small output
(weight gradients) are split along K into up to 32 slices. `scripts/bench/tune.py` writes the
tuning tables from a sweep of every variant over the benchmark workloads.

## Benchmark tools

Properties of `jitnn-bench` (`KernelBenchAccelTest`, `ModelProfileAccelTest`), which
`scripts/bench/run_kernels.sh` and `run_models.sh` set.

| Property | Default | Meaning |
|---|---|---|
| `jitnn.kbench` | `false` | run `KernelBenchAccelTest`: every distinct operator call of the benchmark workloads, timed on the device |
| `jitnn.models` | `false` | run `ModelProfileAccelTest`: whole training and inference steps of the workloads |
| `jitnn.kbench.out` | `kernels.jsonl`, `models.jsonl` | result file, appended |
| `jitnn.kbench.label` | empty | free text recorded with every result line, for example the commits |
| `jitnn.kbench.workloads` | empty | comma-separated workload filter (for example `mlp_train_b256`); empty: all |
| `jitnn.kbench.families` | empty | comma-separated operator families (for example `gemm,convolution`); empty: all |
| `jitnn.kbench.from` | `0` | skip the first n calls, to resume a run |
| `jitnn.kbench.only` | empty | comma-separated call indices to run (1-based, in capture order) |
| `jitnn.kbench.warmup`, `jitnn.kbench.calls` | `20`, `100` | warm-up and minimum timed calls per operator call (timed for at least 0.3 s) |
| `jitnn.kbench.steps` | `30` | timed steps per workload of `ModelProfileAccelTest` |
| `jitnn.kbench.gpu` | `0` | `nvidia-smi` index of the GPU whose clocks are sampled |
| `jitnn.kbench.minMhz` | `1350` | a timed window is retried when the mean SM clock is below this or the GPU was power or thermal capped |
| `jitnn.kbench.bandwidth`, `jitnn.kbench.fp32`, `jitnn.kbench.half` | `1721e9`, `19.0e12`, `245e12` | peak bytes/s and FLOP/s that the efficiency columns are relative to |

## Test properties

Maven properties of the `accel-tests` profile, and system properties read by the tests (pass the
latter through `jitnn.test.extra`).

| Property | Default | Meaning |
|---|---|---|
| `jitnn.test.device` | `0:0` | device for the accelerator tests |
| `jitnn.test.verbose` | `false` | passes `jitnn.verbose` |
| `jitnn.test.profile` | `false` | passes `jitnn.profile` |
| `jitnn.test.gemm` | `auto` | passes `jitnn.gemm` |
| `jitnn.test.extra` | empty | extra JVM arguments, e.g. `-Dtornado.printKernel=True` or `-Djitnn.native=auto` |
| `jitnn.test.deviceMemory` | `12GB` | TornadoVM's device-memory budget (`tornado.device.memory`, whose own default is 4GB) |
| `jitnn.test.keepPlans` | `false` | `CnnAccelTest` keeps the compiled plans between tests instead of closing them |
| `jitnn.mnist` | empty | MNIST directory; the MNIST gate is skipped when empty |
| `jitnn.cifar10` | empty | directory holding `cifar-10-batches-bin`; the CIFAR-10 gates are skipped when empty |
| `jitnn.text` | empty | directory holding `corpus.txt` and the transformer references; the transformer gates are skipped when empty |
| `jitnn.transformer.steps` | 300 | training steps of the transformer gates (must match the PyTorch reference) |
| `jitnn.dist.devices` | empty | comma-separated devices of the collective and data-parallel tests, one rank each, e.g. `0:0,1:0`; empty: two ranks on the test device |
| `jitnn.dist.nccl` | `false` | `DataParallelAccelTest`: also run the collectives and data-parallel training on `NcclCollective` (distinct CUDA devices) |
| `jitnn.bench.dist` | `false` | `DataParallelBenchAccelTest`: training throughput on one device, and on `jitnn.dist.devices` with the host collective and NCCL; `jitnn.bench.dist.model` (`resnet18`, `resnet50`), `.batch`, `.steps`, `.modes` |
| `jitnn.bench.dim`, `.layers`, `.seq`, `.batch` | 128, 4, 128, 16 | model and batch sizes of `TransformerBenchAccelTest` |
| `jitnn.bench.shapes`, `jitnn.bench.tasks` | `false`, empty | with `jitnn.test.profile`, list the device time per GEMM shape, or per task of the operators whose name contains the given text |
| `jitnn.bench.inference` | `false` | run `InferenceBenchAccelTest` (ResNet-50 and BERT-base latency) |
| `jitnn.bench.models` | `resnet50,bert` | the models of `InferenceBenchAccelTest` |
| `jitnn.bench.only` | empty | `InferenceBenchAccelTest`: time only the configurations whose name contains this text (`bert_b32_s128_fp16h`) |
| `jitnn.bench.threads` | `1` | `InferenceBenchAccelTest`: also run this many threads at once and print the samples per second over all of them (with `jitnn.concurrency` slots) |
| `jitnn.bench.split` | `false` | `InferenceBenchAccelTest`: also print where a call's time goes (recording, execution, fetching the output, converting it to floats) |
| `jitnn.bench.profileTasks` | `false` | `InferenceBenchAccelTest` with `jitnn.profile`: also print the device time of every task, not only per operator |
| `jitnn.bench.warmup`, `jitnn.bench.runs` | 10, 50 | warm-up and timed calls of `InferenceBenchAccelTest` |
| `jitnn.bench.out` | empty | directory where `InferenceBenchAccelTest` writes its outputs, for `scripts/bench/t1.py` to compare |
| `jitnn.onnx.t1` | empty | directory with the exported ResNet-50 and BERT-base ONNX files and ONNX Runtime's outputs; `OnnxT1AccelTest` is skipped when empty |
| `jitnn.onnx.t1.match` | empty | run only the ONNX files whose name contains this text |
| `jitnn.probe` | `false` | run `GradProbeTest`: MobileNetV2 step-0 gradients on the reference backend against PyTorch FP64, per parameter (needs `jitnn.mnist`) |
| `jitnn.lstmprobe` | `false` | run `LstmProbeAccelTest`: LSTM training-step times at the shapes of DJL's MNIST LSTM example and two wider ones |
| `jitnn.tileprobe` | `false` | run the CUDA Tile API probes `TileGemmProbeAccelTest` and `TileAttentionProbeAccelTest` |

`NativeAccelTest` runs only when vendor libraries are requested
(`-Djitnn.test.extra=-Djitnn.native=auto`).

TornadoVM's device-memory budget also applies to applications: pass `-Dtornado.device.memory=<N>GB`
for large models. When a plan runs out of it, jitNN closes the other cached plans and recompiles.
