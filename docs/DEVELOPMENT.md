# Development

## Architecture

```
Tensor ops ─▶ Node (op, inputs, outputs, int/float attributes) ─▶ Trace (per thread)
                                                                     │ sync point
                                                                     ▼
       TornadoExecutor: slot table over Storages ─▶ structural key ─▶ plan cache
          miss: Emitter builds one TaskGraph (every op's KernelContext tasks) ─▶ ExecutionPlan
          hit:  rebind inputs and intermediates, replay
```

- **Storage** is the memory behind a tensor and its views. A *persistent* storage (parameters,
  optimiser state) records which compiled plans hold a valid device copy. When another plan needs
  it, jitNN fetches it from a valid plan (`TornadoExecutionResult.transferToHost`) and uploads it
  (`TornadoExecutionPlan.transferToDevice`). *Inputs* are host data, copied into the plan's own
  arrays each execution. *Intermediates* live on the device and are fetched only when read.
- **Plan reuse.** The plan key names each persistent storage it binds. Plans are also indexed by
  structure, without those names: on a key miss, a plan of the same structure whose differing
  persistent storages were all released (`Storage.release`, called when a model closes) is rebound
  to the trace's storages, which take over its buffers. TornadoVM's code cache is per plan, so
  this skips the kernel compilation a new plan would need.
- **Autograd** adds gradient nodes to the same trace, so forward, backward and the optimiser step
  compile together.
- **Reference backend**: `ReferenceExecutor` runs each node eagerly with `Op.reference`.
- **Graph chaining**: a trace longer than `jitnn.graph.maxTasks` tasks is cut at node boundaries
  into several task graphs in one plan. Each array is owned by the first graph that uses it, which
  uploads and persists it; later graphs `consumeFromDevice` it.
- **Rewrites before compilation**: `ops.Fusion` fuses a producer into its only consumer when the
  intermediate is internal (convolution + BatchNorm folding, GELU and ReLU epilogues, add + ReLU).
- **Devices per thread**: each thread records its own trace, and `Engine.onDevice` sets the device
  its plans compile for; the plan cache is keyed by device, so ranks on different devices (or
  backends) share one process.
- **Buffer reuse**: storages created inside `backward()` (except leaf gradients) or inside
  `Grad.inference` are internal; the compiler gives a dead internal buffer to later outputs of the
  same type and size or smaller, and marks it stale after execution.

## Code layout

| Path | Contents |
|---|---|
| `jitnn-core/.../jitnn` | `Tensor`, `Shape`, `DType`, `F`, `Grad`, `Autograd`, `Autocast`, `Linalg`, `Fft`, `SparseMatrix` |
| `jitnn-core/.../engine` | `Engine`, `Trace`, `Node`, `Op`, `Storage`, `TornadoExecutor`, `Emitter`, `Devices` |
| `jitnn-core/.../ops` | one class per operator: emission, host reference, gradient; `Fusion` (inference fusion), `Fp8` |
| `jitnn-core/.../kernels` | the KernelContext kernels; `jitllm/` and `jitrex/` hold kernels copied from those projects |
| `jitnn-core/.../dist` | `Collective`, `HostCollective`, `ProcessGroup` |
| `jitnn-nn/.../nn` | `Module` and the layers, `DataParallel`, `optim/`, `models/` (CNNs, transformers, generative and recurrent models, embedding models), `text/` (`Vocabulary`, `Corpus`) |
| `jitnn-nn/.../ml` | the classical machine-learning estimators, `Lbfgs`, `Binner` and `TreeGrower` |
| `jitnn-nn/.../data` | `Mnist`, `Blobs` |
| `jitnn-onnx/.../onnx` | `Proto` (protobuf reader), `OnnxModel`, `OnnxSession`, `OnnxOps` (one jitNN mapping per ONNX operator), `Rewrites` (load-time pattern rewrites) |
| `scripts/codegen` | GEMM, tensor-core GEMM, simdgroup GEMM, implicit-GEMM convolution, fused attention and multi-tensor optimiser generators (Java has no macros; accumulators must be scalars) |
| `scripts/pytorch` | golden tensors, the MNIST, CNN, transformer, generative and recurrent reference runs, jitNN's Philox in numpy |
| `scripts/sklearn` | scikit-learn references for the classical machine-learning estimators |
| `scripts/onnx` | ONNX goldens (ONNX Runtime outputs) and the ONNX Runtime side of the inference comparison |
| `scripts/bench` | the PyTorch and ONNX Runtime inference comparison, and the same through DJL in `djl/` |

## Adding an operator

Operators outside jitNN use `CustomOp` (see USAGE.md). Built-in operators:

1. Write the kernel in `kernels/` with `KernelContext` (never `@Parallel`). Use `&` and `|`, not
   `&&` and `||`, in conditions.
2. Add an `Op` in `ops/` with `emit` (tasks and grids), `reference` (double precision, written
   independently) and `backward`.
3. Expose it on `Tensor` or `F`.
4. Add a finite-difference check to `GradientTest`, a case to `scripts/pytorch/goldens.py` (then
   regenerate the goldens and map it in `Golden.apply`), and a device check to `OpsAccelTest`.

## Kernels from jitllm

`kernels/jitllm` holds the kernels jitNN takes from jitllm, its LLM inference engine, kept as
they are there apart from the package name: the matrix-vector products on GGUF's Q4_0, Q4_1,
Q4_K, Q5_K and Q6_K blocks (`TransformerComputeKernelsQ4_0` ... `Q6_K`: one work-group per
output row, a warp-shuffle form for CUDA and Metal, and a row-tiled form for small batches of
Q4_0, Q4_1 and Q5_K) and the Q8_0 dequantisation (`NativeProjectionKernels`). jitNN's own
`Q8Kernels` and `MmaGemmKernels` (the 128 x 128 tensor-core GEMM) are adapted from jitllm's.
Only kernels that an operator launches are kept; the rest of jitllm's and jitrex's kernels can be
taken from those repositories when an operator needs them, with goldens and a reference.

## TornadoVM workarounds

Kernels tagged `WORKAROUND(...)` avoid these:

| Issue | Workaround |
|---|---|
| OpenCL crashes on a constant branch Graal proves after nested bounds checks (`LogicConstantNode` in `emitIf`) | Philox uses one thread per element |
| Unsigned shift `>>>` miscompiles (TornadoVM#1140) | Philox's 32x32-bit multiply in `long`, signed shift |
| Sketches are built for the graph's device when a task is added, so `withDevice` is too late for Metal intrinsics | `Devices.pin` sets `<graph>.device` first |
| Metal cannot pass a local-memory array to a helper method | the cross-entropy and LayerNorm/RMSNorm kernels inline their row reductions |
| Graal plugin checks fail under `-ea` on JDK 22+ | the accelerator tests run without assertions |
| A task graph holds at most 256 tasks and 8 KiB of high-level bytecode | traces are chained into several graphs |
| A plan whose execution ran out of memory is not safe to run again | it is dropped and the trace recompiled after evicting other plans |
| Metal's `tanh` returns NaN for large arguments | `ElementwiseKernels.tanh` goes through exp(-2abs(x)) |
| TornadoMath has no erf, erfc, rsqrt, log1p, expm1 | `ElementwiseKernels.erfc`, a Chebyshev fit good to 1.2e-7, for exact GELU; `ElementwiseKernels.log1p` (Goldberg's trick) for softplus and BCE with logits |
| Short-circuit operators inside loops can drop a branch | non-short-circuit AND and OR operators in every kernel |
| TornadoVM inlines a callee only up to 600 IR nodes, counted after the callee's own inlining | the element-wise kernels dispatch to smaller op groups (base, activations, math, NaN-aware), each under the limit; a big helper calling smaller ones does not help |
| The OpenCL backend writes `Float.NaN` as `NaNF`, which does not compile, and specialises scalar kernel arguments into literals (so a NaN or infinity passed as an argument hits the same problem) | the FP8 decoders read NaN and infinity from a small device array |
| `cos` and `sin` fail to compile when their argument becomes a compile-time constant | the DFT reads its phases from a twiddle table computed on the host |

## Vendor libraries

- **Entry point.** Operators reach cuBLAS, cuDNN, cuFFT, cuSPARSE and MLX only through
  `natives.Libs`. Each method returns false when no enabled, available library takes the call;
  the operator then emits its own kernels.
- **Library classes.** `natives.Cuda` and `natives.Mlx` hold the `TaskGraph.libraryTask` calls.
  They load only after `engine.Native` has probed the library on the device, so jitNN runs without
  TornadoVM's library modules.
- **Emitting a call.** `Emitter.library(lib, function, adder)` adds a task without a launch grid.
  The function name (`"cudnn.conv"`) is the unit that is turned off when a plan fails.
- **Unread outputs.** `Emitter.unread(tensor)` tells an operator that nothing reads an output, such
  as a norm's statistics or attention's log-sum-exp in inference. It may then call a library that
  does not produce it.
