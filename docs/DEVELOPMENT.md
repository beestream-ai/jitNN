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
- **Autograd** adds gradient nodes to the same trace, so forward, backward and the optimiser step
  compile together.
- **Reference backend**: `ReferenceExecutor` runs each node eagerly with `Op.reference`.
- **Graph chaining**: a trace longer than `jitnn.graph.maxTasks` tasks is cut at node boundaries
  into several task graphs in one plan. Each array is owned by the first graph that uses it, which
  uploads and persists it; later graphs `consumeFromDevice` it.
- **Buffer reuse**: storages created inside `backward()` (except leaf gradients) or inside
  `Grad.inference` are internal; the compiler gives a dead internal buffer to later outputs of the
  same type and size or smaller, and marks it stale after execution.

## Code layout

| Path | Contents |
|---|---|
| `jitnn-core/.../jitnn` | `Tensor`, `Shape`, `DType`, `F`, `Grad`, `Autograd` |
| `jitnn-core/.../engine` | `Engine`, `Trace`, `Node`, `Op`, `Storage`, `TornadoExecutor`, `Emitter`, `Devices` |
| `jitnn-core/.../ops` | one class per operator: emission, host reference, gradient |
| `jitnn-core/.../kernels` | the KernelContext kernels |
| `jitnn-nn/.../nn` | `Module`, `Linear`, `ReLU`, `Sequential`, `optim/` |
| `jitnn-nn/.../data` | `Mnist`, `Blobs` |
| `scripts/codegen` | GEMM, simdgroup GEMM and implicit-GEMM convolution generators (Java has no macros; accumulators must be scalars) |
| `scripts/pytorch` | golden tensors, the MNIST reference run, jitNN's Philox in numpy |

## Adding an operator

1. Write the kernel in `kernels/` with `KernelContext` (never `@Parallel`). Use `&` and `|`, not
   `&&` and `||`, in conditions.
2. Add an `Op` in `ops/` with `emit` (tasks and grids), `reference` (double precision, written
   independently) and `backward`.
3. Expose it on `Tensor` or `F`.
4. Add a finite-difference check to `GradientTest`, a case to `scripts/pytorch/goldens.py` (then
   regenerate the goldens and map it in `Golden.apply`), and a device check to `OpsAccelTest`.

## Kernels copied from jitllm and jitrex

`kernels/jitllm` holds the GPU kernels of jitllm (the LLM inference engine: GEMV and tensor-core
GEMMs, quantised Q4/Q5/Q6/Q8 matmuls, paged and flash attention for prefill and decode, RMSNorm,
RoPE, SwiGLU/GeGLU, MoE routing and experts, Gated DeltaNet), and `kernels/jitrex` holds those of
jitrex (TiRex-2: sLSTM and mLSTM recurrences, variate attention, the portable FP16 GEMM and a CUDA
Tile API GEMM). They are forward-only inference kernels, kept as they are in their source
repositories apart from the package name and their few `@Parallel` loops, which were turned into
KernelContext kernels. An operator adopts one by wrapping it and adding goldens and a reference;
so far `MmaGemmKernels` (jitllm's 128 x 128 tensor-core GEMM, with a bfloat16 twin and a
strided variant that serves every transpose and split-K) is in use.

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
