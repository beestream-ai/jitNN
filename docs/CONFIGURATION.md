# Configuration

## Runtime options

System properties, read once at start-up.

| Property | Default | Meaning |
|---|---|---|
| `jitnn.backend` | `tornado` when a TornadoVM runtime is present, else `reference` | where operators run |
| `jitnn.device` | TornadoVM's default device | `<driver>:<device>`, as listed by `tornado --devices` |
| `jitnn.plan.cache` | `16` | compiled plans kept; the least recently used is closed, saving parameters first |
| `jitnn.cudaGraph` | `true` | on CUDA, capture each plan as a CUDA graph and replay it |
| `jitnn.verbose` | `false` | log every compiled plan: device, nodes, tasks, build time |
| `jitnn.printKernels` | `false` | print each compiled plan's operators and generated kernel source: `true` for standard output, or a file name (see `Engine.printKernels`) |
| `jitnn.profile` | `false` | enable TornadoVM's profiler: `Engine.lastProfile()` gives the last step's kernel time, `Engine.kernelBreakdown()` the device time per operator task (GEMMs labelled with their shape); on CUDA also set `jitnn.cudaGraph=false`, as the profiler reports no kernel time under CUDA graphs |
| `jitnn.gemm` | `auto` | force a GEMM kernel: `simdgroup`, `128`, `64`, `16` |
| `jitnn.transpose` | `32` | transpose kernel: `32` (32x32 tiles), `16`, or `naive` |
| `jitnn.graph.maxTasks` | `40` | tasks per TornadoVM task graph; longer traces become a chain of graphs (a graph holds at most 256 tasks and 8 KiB of bytecode) |
| `jitnn.reuseBuffers` | `true` | reuse dead backward-pass and inference buffers inside a compiled step |
| `jitnn.splitK` | `true` | split the reduction of GEMMs and convolutions whose output fills too few work-groups (A^T B weight gradients, and forward products), summing the partials in order |
| `jitnn.splitK.groups` | `256` | work-groups a forward product should reach before it stops splitting K (about two per SM on an A100) |
| `jitnn.fuse` | `true` | inference fusion: convolution + BatchNorm (+ ReLU) on folded weights, GELU in the GEMM epilogue, add + ReLU in one kernel |
| `jitnn.onnx.rewrite` | `true` | `jitnn-onnx`: rewrite MatMul by a constant (+ bias) into linear layers, the erf GELU chain into GELU and attention chains into one attention operator at load |
| `jitnn.onnx.fusedAttentionRows` | `32` | `jitnn-onnx`: rewritten attention uses the fused kernel when batch x heads is at most this, else batched GEMMs and a fused scale-softmax |
| `jitnn.tensorCores` | `true` | on CUDA, half-precision products (FP16, BF16 operands) run on tensor cores (`mma.sync` m16n8k16, compute capability 8.0+) |
| `jitnn.fusedAttention` | `true` | FP32 attention with head dimensions up to 128 runs the fused FlashAttention-style kernels; `false` composes batched GEMMs and a softmax |

## GEMM selection (`auto`)

| Condition | Kernel |
|---|---|
| Metal, M and N multiples of 8, output at least 64x64 | `gemmSimdgroup`: simdgroup 8x8 matrix units, 64x64 per 128 threads; the bias/ReLU epilogue runs as a second kernel |
| M, N at least 1024 and K at least 64 | `gemm128`: 128x128 per 256 threads, 8x8 outputs per thread |
| output at least 64x64 and K at least 16 | `gemm64`: 64x64 per 256 threads, 4x4 outputs per thread |
| otherwise | `gemm`: 16x16 tiles, one output per thread |

All variants take both transposes and an epilogue (bias, ReLU) without materialising anything.
Half-precision operands (FP16, BF16) use the same choice with kernels that convert on load: the
simdgroup kernels on Metal, the 64x64 register-blocked ones elsewhere. A^T B products with a long
reduction and a small output (weight gradients) are split along K into up to 32 slices.

## Test properties

Maven properties of the `accel-tests` profile.

| Property | Default | Meaning |
|---|---|---|
| `jitnn.test.device` | `0:0` | device for the accelerator tests |
| `jitnn.test.verbose` | `false` | passes `jitnn.verbose` |
| `jitnn.test.profile` | `false` | passes `jitnn.profile` |
| `jitnn.test.gemm` | `auto` | passes `jitnn.gemm` |
| `jitnn.mnist` | empty | MNIST directory; the MNIST gate is skipped when empty |
| `jitnn.cifar10` | empty | directory holding `cifar-10-batches-bin`; the CIFAR-10 gates are skipped when empty |
| `jitnn.text` | empty | directory holding `corpus.txt` and the transformer references; the transformer gates are skipped when empty |
| `jitnn.bench.dim`, `.layers`, `.seq`, `.batch` | 128, 4, 128, 16 | model and batch sizes of `TransformerBenchAccelTest` |
| `jitnn.bench.shapes`, `jitnn.bench.tasks` | `false`, empty | with `jitnn.test.profile`, list the device time per GEMM shape, or per task of the operators whose name contains the given text |
| `jitnn.transformer.steps` | 300 | training steps of the transformer gates (must match the PyTorch reference) |
| `jitnn.dist.devices` | empty | comma-separated devices of the collective and data-parallel tests, one rank each, e.g. `0:0,1:0`; empty: two ranks on the test device |
| `jitnn.bench.inference` | `false` | run `InferenceBenchAccelTest` (ResNet-50 and BERT-base latency) |
| `jitnn.bench.models` | `resnet50,bert` | the models of `InferenceBenchAccelTest` |
| `jitnn.bench.warmup`, `jitnn.bench.runs` | 10, 50 | warm-up and timed calls of `InferenceBenchAccelTest` |
| `jitnn.bench.out` | empty | directory where `InferenceBenchAccelTest` writes its outputs, for `scripts/bench/t1.py` to compare |
| `jitnn.onnx.t1` | empty | directory with the exported ResNet-50 and BERT-base ONNX files and ONNX Runtime's outputs; `OnnxT1AccelTest` is skipped when empty |
| `jitnn.onnx.t1.match` | empty | run only the ONNX files whose name contains this text |
| `jitnn.test.extra` | empty | extra JVM arguments, e.g. `-Dtornado.printKernel=True` |
| `jitnn.test.deviceMemory` | `12GB` | TornadoVM's device-memory budget (`tornado.device.memory`, whose own default is 4GB) |

TornadoVM's device-memory budget also applies to applications: pass `-Dtornado.device.memory=<N>GB`
for large models. When a plan runs out of it, jitNN closes the other cached plans and recompiles.
