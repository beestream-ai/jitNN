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
- **OpenCL on the host CPU.** When TornadoVM can see an OpenCL device of type CPU (PoCL, the
  Intel CPU runtime), operators without a CPU kernel run there instead of their reference, with
  the kernels jitNN uses on GPUs, when that is faster. Each operator and shape runs its reference
  twice (the first warms the JIT) and on the device three times (the first compiles), timed with
  the copies, and keeps the faster place: the rotary embedding of a small LLaMA was slower on the device than its
  reference, the U-Net operators much faster. Consecutive operators that chose the device form
  one TornadoVM plan. Its inputs are copied in and the outputs a later node reads come back to
  host arrays, so the two sides never share memory. An operator the device fails to compile runs
  its reference from then on.
  `-Djitnn.cpu.opencl` picks the device or turns this off; the counters show calls per operator
  that ran there.
  The CPU drivers install signal handlers that take the signals the JVM raises on purpose
  (safepoint polls, implicit null checks), so the JVM dies of a SIGSEGV with no error report at a
  random point. Preloading the JVM's signal chaining library,
  `LD_PRELOAD=$JAVA_HOME/lib/libjsig.so`, fixes it for PoCL and for the Intel CPU runtime
  (`POCL_SIGFPE_HANDLER=0` alone does not). Without it the device is left alone, with a warning.
  The drivers start one worker thread per logical CPU, and those keep competing with the backend's
  threads after a plan returns: the GEMMs of a LLaMA step ran 10% slower on PoCL, 50% slower on the
  Intel runtime. Give the driver the backend's thread count: `POCL_MAX_PTHREAD_COUNT` (PoCL 3) or
  `POCL_CPU_MAX_CU_COUNT` (later PoCL versions), `CL_CONFIG_CPU_TBB_NUM_WORKERS` (the Intel runtime).
  On the x86 machine with 32 threads (ms per training step, mean of three medians of 10 steps; each
  runtime with its own reference, measured in the same session):

  | Runtime | Workload | reference | timed choice |
  |---|---|---|---|
  | PoCL 3.0 | U-Net b32 | 653 | 155 |
  | PoCL 3.0 | LLaMA b4 s256 | 1248 | 1134 |
  | Intel CPU runtime 2026.2 | U-Net b32 | 644 | 134 |
  | Intel CPU runtime 2026.2 | LLaMA b4 s256 | 1154 | 1074 |

  The U-Net moves its transposed convolutions, upsampling and its backward, large concat and narrow,
  and max-pool backward to the device; LLaMA its rotary embedding. Without the thread cap the Intel
  runtime made LLaMA slower than its reference (1719 ms). MobileNetV2, the VAE and the MLP have no
  operator without a CPU kernel and do not change.
  Running whole models on the OpenCL device instead does not pay overall: on PoCL GPU-shaped kernels
  ran ResNet-50 and BERT 4–15x slower than the Vector API kernels, on the Intel runtime they come
  close and win two of eight rows (see Measuring).
- **Native libraries.** Linear, matmul and convolutions (forward with bias and ReLU, backward data
  and weights) can also run on native libraries, called through the foreign function API with no
  native code in jitNN: a BLAS's `cblas_sgemm` (`jitnn.cpu.blas`: MKL, OpenBLAS, Arm Performance
  Libraries, Apple Accelerate), and for convolutions oneDNN (`jitnn.cpu.onednn`) and Apple's BNNS
  (`jitnn.cpu.bnns`, macOS). A BLAS convolution is im2col and one GEMM per image (forward), or per
  group of images side by side (backward, then col2im for the data gradient). oneDNN and BNNS keep
  one plan per convolution shape in the layout they want; a weight is reordered or packed again
  only when its version changes. Each operator and shape takes the available ways in turn, three calls
  each (the first warms the JIT or builds the plan, the best of the other two counts), and runs
  the fastest; those first calls fall in the program's warm-up, so after 64 more calls the ways
  are timed once more, and that choice is final (timing only during warm-up picked the slower way
  for BERT's attention in some runs). A library that rejects a shape leaves it to the kernel.
  Attention can run on the BLAS too: each head's two products on the library, the softmax rows of
  all heads in parallel between them (BERT b1 on an M4 Pro: attention 0.33 to 0.20 ms per layer,
  the model 24.4 to 22.7 ms). Libraries use the backend's thread count
  (MKL and OpenBLAS by their calls, OpenMP builds of oneDNN and Arm Performance Libraries per
  calling thread). Not solved yet: the libraries' idle worker threads keep spinning after a call and
  compete with the backend's own threads, which the per-operator timing does not see. On the x86
  machine MKL made ResNet-50 b32 slower end to end (258 to 418 ms) although it won each GEMM it took,
  and MKL together with oneDNN (two OpenMP runtimes) slower still; OpenBLAS on the Arm machine
  slowed training by 20–55%. oneDNN alone helped convolutional networks (ms, training ms per step,
  one round):

  | Workload | kernels | oneDNN |
  |---|---|---|
  | ResNet-50 b32 | 258 | 221 |
  | ResNet-50 (CIFAR) training b32 | 485 | 449 |
  | MobileNetV2 training b64 | 1543 | 301 |

  On an Apple M4 Pro, Accelerate's GEMM (on the AMX units) runs at 3.0–3.5 TFLOP/s against 0.5 for
  the Vector API kernel (4 FP32 lanes on NEON), and its threads do not get in the way (ms, training
  ms per step, 12 threads unless noted, single runs):

  | Workload | kernels | Accelerate (8 threads) | Accelerate | Accelerate + BNNS |
  |---|---|---|---|---|
  | ResNet-50 b1 | 40.5 | 16.3 | 18.1–22.0 | 15.6 |
  | BERT b1 s128 | 72 | 28 | 29 | |
  | ResNet-50 b32 | 677 | 412 | 438–448 | 256 |
  | BERT b32 s128 | 1679 | 433 | 452 | |
  | BERT training b8 s128 | 1713 | 581 | 608 | |
  | ResNet-18 training b64 | | | 536 | 267 |
  | MobileNetV2 training b64 | | | 1394 | 657 |

  These runs had the pool spinning 200 µs; without it (now the macOS default) ResNet-50 b1 runs
  at 12.3–13.2 ms, BERT b1 at 23.6–24.9 and BERT training at 541–551 ms per step.
  The BNNS filter API is deprecated since macOS 15 in favour of BNNS graphs (which need a compiled
  Core ML model) but still runs; its backward pass needs filters that read the weights in place.
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
  64. Idle threads spin for `jitnn.cpu.spinMicros` before sleeping (200 µs; none on macOS, where spinning made ResNet-50 b1 29% slower on the Vector API kernels and BERT b1 16% slower with Accelerate on an M4 Pro).
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
- **OpenCL on the CPU, whole models.** The TORNADO backend on an OpenCL CPU device (the x86
  machine, FP32, median ms; training in ms per step) against the Vector API kernels of the same
  session. PoCL 3.0 in one session; the Intel CPU runtime 2026.2 with all its threads (capped at 32
  it was 1.3–2x slower) in another, mean of two rounds (GPT-2: one):

  | Workload | PoCL | Vector API | Intel runtime | Vector API |
  |---|---|---|---|---|
  | ResNet-50 b1 | 302 | 21 | 29.6 | 19.8 |
  | BERT b1 s128 | 447 | 41 | 35.0 | 31.1 |
  | ResNet-50 b32 | 1186 | 261 | **255** | 277 |
  | BERT b32 s128 | 2888 | 419 | 490 | 395 |
  | ResNet-18 training b64 | 1338 | 350 | **260** | 341 |
  | ResNet-50 (CIFAR) training b32 | 2644 | 508 | 492 | 472 |
  | BERT training b8 s128 | 7221 | 747 | 838 | 732 |
  | GPT-2 training b8 s256 | 15350 | 1409 | 1549 | 1381 |

  The Intel runtime's compiler vectorises the GPU kernels well enough to beat the Vector API on
  batched convolutions (ResNet-50 b32, ResNet-18 training); PoCL's does not. PoCL passed 452 of the
  490 device tests; the rest need FP16, which its CPU driver lacks. Both drivers need `libjsig`
  (see Operator dispatch): without it the Intel runtime crashed the JVM on the first kernel launch,
  in TornadoVM's own tests too.
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
