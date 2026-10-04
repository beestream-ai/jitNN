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
| `jitnn.profile` | `false` | enable TornadoVM's profiler: `Engine.lastProfile()` gives the last step's kernel time, `Engine.kernelBreakdown()` the device time per operator task (GEMMs labelled with their shape); on CUDA also set `jitnn.cudaGraph=false`, as the profiler reports no kernel time under CUDA graphs |
| `jitnn.gemm` | `auto` | force a GEMM kernel: `simdgroup`, `128`, `64`, `16` |
| `jitnn.transpose` | `32` | transpose kernel: `32` (32x32 tiles), `16`, or `naive` |
| `jitnn.graph.maxTasks` | `40` | tasks per TornadoVM task graph; longer traces become a chain of graphs (a graph holds at most 256 tasks and 8 KiB of bytecode) |
| `jitnn.reuseBuffers` | `true` | reuse dead backward-pass and inference buffers inside a compiled step |
| `jitnn.splitK` | `true` | split the reduction of A^T B GEMMs (weight gradients) whose output fills too few work-groups, summing the partials in order |
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
| `jitnn.test.extra` | empty | extra JVM arguments, e.g. `-Dtornado.printKernel=True` |
| `jitnn.test.deviceMemory` | `12GB` | TornadoVM's device-memory budget (`tornado.device.memory`, whose own default is 4GB) |

TornadoVM's device-memory budget also applies to applications: pass `-Dtornado.device.memory=<N>GB`
for large models. When a plan runs out of it, jitNN closes the other cached plans and recompiles.
