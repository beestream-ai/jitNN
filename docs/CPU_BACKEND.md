# CPU backend (Java Vector API): design

Status: phases 1–4 done (executor, GEMM and convolution, inference operators, training kernels and buffer reuse); remaining: tuning (phase 5).

## Goal

Run jitNN models fast on CPUs with no GPU and no native libraries. The backend is pure Java, FP32,
multithreaded, and vectorised with `jdk.incubator.vector`. It runs on any JDK 22+ and on x86
(AVX2, AVX-512) and AArch64 (NEON, SVE).

Today the only CPU path is `REFERENCE`. It runs scalar, single-threaded, double-accumulating
reference code, one node at a time. It is an oracle, not a backend: ResNet-50 at batch 1 takes
about 30 s per call on a 64-core Xeon, while PyTorch takes 25 ms.

## Where it plugs in

A third value `Engine.Backend.CPU`, selected with `-Djitnn.backend=cpu`.

- **Default.** TORNADO when a TornadoVM runtime is on the class path, otherwise CPU.
  REFERENCE stays available for tests and as the oracle.
- **Execution model: lazy, like TORNADO.** `Engine.record` adds nodes to the trace. At `Engine.sync`,
  `CpuExecutor.run(nodes, wanted)` does the following:
  1. Applies `Fusion.rewrite(nodes, wanted)`, the same pass the GPU path uses. Conv, BatchNorm
     and ReLU fold into one conv with an epilogue; linear and GELU fold into one GEMM epilogue;
     add and ReLU fold into one element-wise op.
  2. Runs the nodes in program order.
  3. Releases intermediate buffers after their last use (liveness over the trace).
- **No compile step and no plan cache.** A trace on the CPU costs only the fusion pass. The
  trace structure can be cached later if the pass shows up in profiles.

## Operator dispatch

`Op` gains one default method:

```java
/** Computes {@code node} with the CPU backend's kernels; false when the operator has none (the executor then runs {@link #reference}). */
default boolean cpu(Node node) {
    return false;
}
```

- **Fallback.** Every one of the ~120 operators works on day one through `reference`. The
  executor counts fallbacks per operator name. `-Djitnn.cpu.stats=true` prints them at exit, so
  missing kernels are visible.
- **Code split.** Operators decode their own `iattr` layout (the constants are private to each
  op class). They then call shared primitives in a new package `org.beehive.jitnn.cpu`. Those
  primitives take only segments, offsets and sizes. That package is the only code that imports
  `jdk.incubator.vector`.

## Data

- **Storage and access.** Storage stays FP32 in TornadoVM `FloatArray` (off-heap). Kernels read
  and write through `FloatArray.getSegment()` with
  `FloatVector.fromMemorySegment(SPECIES, seg, off, LITTLE_ENDIAN)`, with no copies.
  `Tensor.wrap` heap arrays use `FloatVector.fromArray`.
- **Vector width.** Width is `FloatVector.SPECIES_PREFERRED`: 16 lanes on AVX-512, 8 on AVX2,
  4 on NEON. Tails use masks.
- **Inputs in other types.** FP16 inputs are converted to FP32 at the node boundary. INT8 and
  Q8_0 are later work.

## Threading

- **Pool.** One process-wide pool of `jitnn.cpu.threads` platform threads. The default is the
  physical core count, capped at one socket's cores: hyper-threads do not help FMA-bound GEMM,
  and on a two-socket Xeon one socket's 32 cores ran ResNet-50 and BERT 10–40% faster than all
  64. Idle threads spin for `jitnn.cpu.spinMicros` before sleeping.
- **Parallel-for.** `Cpu.parallelFor(n, grain, body)` uses static contiguous chunks. There is no
  work stealing, so per-call overhead stays near the cost of one wake-up and one barrier.
- **Grain.** Each op picks the grain from its work estimate. Below about 32 K FLOPs it runs on
  the calling thread.
- **Pool ownership.** Calls from several user threads share the pool. Each sync runs
  single-threaded if the pool is busy, rather than oversubscribing.

## Kernels, in priority order

Priority follows where ResNet-50 and BERT spend time on the GPU path, and what the baseline
measures.

1. **GEMM: `cpu.Gemm.sgemm`.**
   - Structure: BLIS-style. B is packed into KC×NR panels and A into MC×KC blocks. The
     register-tiled micro-kernel computes MR×NR with `fma`: 6×(2·VLEN) on AVX-512 and AVX2,
     8×(2·VLEN) on NEON. Blocking constants come from the vector width and are tuned once per
     machine class.
   - Parameters: it covers `ta`/`tb`, batch with strides, and an epilogue (bias, ReLU, GELU,
     add).
   - Threading: it parallelises over (batch, M blocks, N blocks).
   - Users: `linear`, `matmul`, conv and attention.
2. **Conv2d (NCHW).**
   - 1×1 stride-1 convolutions are a plain GEMM: `W[Cout,Cin] · X[Cin,H·W]`.
   - Other convolutions use im2col per image, or per row band when the column buffer exceeds
     L2, followed by GEMM.
   - Depthwise and grouped convolutions with few channels per group use a direct, vectorised
     kernel over W.
   - The fused epilogue applies bias, folded BatchNorm and ReLU.
3. **Element-wise.**
   - Ops: `binary`, `unary`, `fused`, `cast`.
   - Fast paths cover contiguous inputs, scalar broadcast and row or channel broadcast.
   - Transcendentals use `VectorOperators.EXP`, `TANH` and `LOG`: SVML intrinsics on x86, Java
     fallback elsewhere. AArch64 speed needs measuring.
4. **Row ops.** `softmax`, `attention_softmax`, `row_norm` (LayerNorm, RMSNorm), `reduce`,
   `cross_entropy`, `nll`. One pass per row computes max or sum; a vector loop then normalises.
5. **Spatial.** `pool2d`, `adaptive_pool`, `batch_norm` (inference, and training statistics),
   `pad`, `upsample`.
6. **Data movement.** `permute`, `narrow`, `concat`, `gather`, `index_select`, `embedding`.
   These use blocked copies; the inner contiguous dimension uses `MemorySegment.copy`.
7. **Attention: `flash_attention`.** Per (batch, head) and Q block: GEMM QKᵀ, then row softmax,
   then GEMM ·V, with a tiled online softmax so memory stays O(seq·block).
8. **Training.**
   - The backward kernels of the above: `conv2d_bwd_data` is GEMM plus col2im, and
     `conv2d_bwd_weight` is im2colᵀ GEMM.
   - Plus `batch_norm_bwd`, `row_norm_bwd`, `softmax_bwd`, `pool2d_bwd`, `embedding_bwd` and
     `optim` (SGD, Adam, AdamW).

Steps 1–6 cover ResNet-50 and BERT inference with no fallbacks. Step 8 covers the training
benchmark.

## Accuracy

The backend uses FP32 storage and FP32 accumulation. GEMM sums in blocks of KC, so error grows
with log(K), not K.

Acceptance against REFERENCE uses the same per-op tolerances as the GPU golden tests. The
existing `GoldenTest` and `ReferenceOpsTest` gain a CPU run:

- every op is checked against REFERENCE;
- sizes include tails that are not a multiple of the vector width;
- strided and transposed GEMM layouts are included.

Model level: logits match PyTorch FP32 within the bounds that t1 already reports.

## Build and run

- **Module flag.** `jdk.incubator.vector` needs `--add-modules jdk.incubator.vector` at compile
  and run time. The flag goes into the compiler args and the surefire `argLine`.
- **Runtime classes.** The CPU backend needs no TornadoVM runtime and no driver, only the
  `tornado-api` jar for `FloatArray`.
- **Module missing.** If the module is absent at run time, `Backend.CPU` fails fast with a
  message naming the flag. It does not degrade silently to scalar code.

## Measuring

- **Baseline.** The baseline was captured before any code (2026-10-28) on two machines:
  - an x86 machine: Xeon Gold 6338, 2×32 cores, AVX-512;
  - an AArch64 machine: Neoverse-N1, 128 cores, NEON.

  Each machine has PyTorch CPU and ONNX Runtime CPU numbers, on all threads and on 1 thread, for
  ResNet-50 (b1/b32/b128), BERT (b1/b32 s128, b8 s512), training, and the jitNN REFERENCE
  backend.
- **Same harness.** `t1.py --device cpu` and `InferenceBenchAccelTest` with
  `-Djitnn.backend=cpu` produce the comparable rows.
- **Per-phase checks.** Each phase reports:
  - GEMM GFLOP/s against machine peak;
  - per-op time against REFERENCE;
  - end-to-end time against PyTorch and ORT, single-thread and all-thread.

## Phases

1. **Skeleton.**
   - `Backend.CPU` and `CpuExecutor` (trace, fusion, liveness, fallback).
   - The pool and fallback statistics.
   - The CPU-vs-REFERENCE test harness.
2. **GEMM.** GEMM with epilogues, plus `linear`, `matmul`, and conv 1×1 and im2col. Measure
   GFLOP/s.
3. **Remaining inference ops.** Element-wise, row, spatial and movement ops, plus flash
   attention. Done when ResNet-50 and BERT inference run with zero fallbacks.
4. **Training.** Backward kernels and the optimiser, then the training benchmark.
5. **Memory and tuning.**
   - Reuse intermediate buffers across syncs (HostPool).
   - Per-machine blocking constants, chosen with the autotuner in `Autotune`.
   - Optional NHWC convs.

After this backend: option 2 (TornadoVM OpenCL on a CPU device) and option 3 (native
BLAS/oneDNN via FFM) are measured against it on the same rows.

## Open questions

- Default thread count: physical cores (proposed) or `availableProcessors()`.
- Whether CPU should replace REFERENCE as the default when no TornadoVM runtime is present
  (proposed: yes).
- Whether the GPU path should also use the CPU kernels for its host-side work, such as weight
  folding on the host. Out of scope for now.
