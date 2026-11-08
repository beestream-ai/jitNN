# CPU backend

The CPU backend runs jitNN models on the host CPU, with no GPU and no TornadoVM runtime. Its
kernels are pure Java: FP32, multithreaded, and vectorised with the Java Vector API
(`jdk.incubator.vector`) on x86 (AVX2, AVX-512) and AArch64 (NEON). Native libraries (a BLAS,
oneDNN, Apple's BNNS) can take some operators when they are installed.

## Selecting it

```bash
java --add-modules jdk.incubator.vector -Djitnn.backend=cpu ...
```

- `jitnn.backend` takes `tornado`, `cpu` or `reference`. Without it, jitNN uses `tornado` when a
  TornadoVM runtime is on the class path, else `cpu` when the JVM has the Vector API module, else
  `reference`.
- Without `--add-modules jdk.incubator.vector`, `-Djitnn.backend=cpu` fails at once with a message
  naming the flag; it does not fall back to scalar code.
- The backend needs only the `tornado-api` jar (for `FloatArray`), no TornadoVM runtime or driver.
- `reference` runs each operator eagerly with its Java reference implementation. Tests use it as
  the oracle.

## How it runs

Operators are recorded lazily, as on the TornadoVM backend. At each sync point (`JitnnRuntime.sync`,
`t.item()`, ...) the backend:

1. rewrites the trace with the same fusion pass as on a GPU (for example convolution, BatchNorm and
   ReLU become one convolution with an epilogue);
2. runs the nodes in program order;
3. returns each intermediate to the host pool after its last use.

There is no compile step. When a trace of the same structure runs again (the next training or
inference step), its intermediates reuse the arrays of the previous run, as a compiled plan reuses
its device buffers. Tensors not requested at that sync are then gone, as on a device.

## What runs where

| Where | Operators |
|---|---|
| Native library, when installed and faster for that shape | `linear`, `matmul` (BLAS); convolution forward with bias and ReLU, backward data and backward weights (BLAS as im2col and GEMM, oneDNN, BNNS); attention (BLAS for the two products per head) |
| Vector API kernel | `linear`, `matmul`, convolution and both its gradients, element-wise (`binary`, `unary`, fused), `softmax`, `row_norm` (LayerNorm, RMSNorm), `reduce`, `cross_entropy`, `batch_norm`, `pool2d`, `embedding`, `flash_attention`, the backward passes of these, the optimiser step |
| OpenCL device on the host CPU | operators with no CPU kernel, when such a device is set up and faster (see below) |
| Reference implementation | everything else |

`-Djitnn.cpu.stats=true` prints at exit the calls, time and reference fallbacks per operator, so
missing kernels are visible. `JitnnRuntime.cpuStats()` returns the counts; `JitnnRuntime.cpuReport()` returns
the table since its last call.

### Native libraries

The libraries are called through the Java foreign function API; jitNN ships no native code.

| Option | Library | `auto` (default) |
|---|---|---|
| `jitnn.cpu.blas` | `cblas_sgemm` of Intel MKL, OpenBLAS, Arm Performance Libraries or Apple Accelerate | the first found on the library path |
| `jitnn.cpu.onednn` | oneDNN (`libdnnl`) | found on the library path |
| `jitnn.cpu.bnns` | Apple BNNS (Accelerate), convolutions including grouped | used on macOS |

Each takes `auto`, `off` or a library file (BNNS: `auto` or `off`).

- **Choice per shape.** Each operator and shape tries the available ways (kernel, BLAS, oneDNN,
  BNNS), three calls each, and keeps the fastest. After 64 more calls it times them once more, and
  that choice is final. `-Djitnn.cpu.native.choose=kernel|blas|onednn|bnns` forces one way where it
  is available.
- **Rejected shapes.** A library that rejects a shape leaves it to the kernel from then on.
- **Threads.** The libraries use the backend's thread count. Use an OpenMP build of oneDNN: jitNN
  sets its thread count, while a TBB build takes every core. Calls from inside a parallel loop use
  the kernel only.
- **Plans and weights.** oneDNN and BNNS keep one plan per convolution shape in the layout they
  want. A weight is reordered or packed again only when it changes.
- **Idle library threads.** Some libraries' worker threads keep spinning after a call and compete
  with the backend's threads, which the per-shape timing does not see. If a model runs slower with
  a library, turn it off.
- **BNNS.** Its filter API is deprecated since macOS 15 but still runs.

### OpenCL on the host CPU

When a TornadoVM runtime is present and it sees an OpenCL device of type CPU (PoCL, the Intel CPU
runtime), operators without a CPU kernel can run there with jitNN's GPU kernels.

- **Choice.** Each operator and shape runs its reference twice and on the device three times,
  timed with the copies, and keeps the faster place. `-Djitnn.cpu.opencl.choose=always` sends it to
  the device without timing.
- **Plans.** Consecutive operators that chose the device run as one TornadoVM plan. Inputs are
  copied in and outputs a later node reads are copied back; the two sides never share memory.
- **Failures.** An operator the device cannot compile (FP16 without `cl_khr_fp16`, say) runs its
  reference from then on.
- **Signal chaining (required).** The CPU OpenCL drivers install signal handlers that take signals
  the JVM raises on purpose, and the JVM then dies of a SIGSEGV at a random point. Preload the JVM's
  signal chaining library: `LD_PRELOAD=$JAVA_HOME/lib/libjsig.so`. Without it `auto` leaves the
  device alone and prints a warning.
- **Driver threads.** Give the driver the backend's thread count, or its idle threads compete with
  the backend's: `POCL_MAX_PTHREAD_COUNT` (PoCL 3), `POCL_CPU_MAX_CU_COUNT` (later PoCL),
  `CL_CONFIG_CPU_TBB_NUM_WORKERS` (Intel runtime).

`JitnnRuntime.cpuOpenClDevice()` returns the device in use, or null.

## Threads

One process-wide pool runs the parallel loops in static contiguous chunks; the calling thread takes
the first.

- **Size.** By default the physical cores the process may use, at most one socket's; on Apple
  silicon the performance cores only.
- **Several user threads.** A loop started while another thread holds the pool, or from inside a
  parallel loop, runs on the calling thread alone rather than oversubscribing the cores.
- **Spinning.** Idle pool threads, and a caller waiting for them, spin briefly before they sleep,
  so the operators of one step hand work on without waking threads up.

## Options

| Option | Default | Effect |
|---|---|---|
| `jitnn.backend` | see above | `tornado`, `cpu` or `reference` |
| `jitnn.cpu.threads` | physical cores, at most one socket (Apple silicon: performance cores) | threads of the parallel loops, the calling thread included |
| `jitnn.cpu.spinMicros` | `200` (`0` on macOS) | how long idle threads spin before they sleep |
| `jitnn.cpu.poolBytes` | 2 GiB | host arrays kept for reuse, unless `jitnn.hostPool.bytes` is set |
| `jitnn.cpu.reuse` | `true` | reuse the previous run's arrays for a trace of the same structure |
| `jitnn.cpu.reuse.traces` | `32` | trace structures remembered per thread for that reuse |
| `jitnn.cpu.gemm.kc`, `.mcPanels`, `.nc`, `.tileMacs` | `256`, `16`, `256`, `1048576` | GEMM blocking: depth per block, rows of A per tile (in panels), columns of B per tile, minimum multiply-adds per parallel tile |
| `jitnn.cpu.blas`, `jitnn.cpu.onednn`, `jitnn.cpu.bnns` | `auto` | native libraries (see above) |
| `jitnn.cpu.native.choose` | `measure` | `kernel`, `blas`, `onednn` or `bnns` forces that way |
| `jitnn.cpu.opencl` | `auto` | OpenCL CPU device for operators without a kernel: `auto`, `off` or `<driver>:<device>` |
| `jitnn.cpu.opencl.choose` | `measure` | `always` skips the timing |
| `jitnn.cpu.stats` | `false` | time every operator and print the table at exit |
| `jitnn.cpu.stats.shapes` | `false` | with `jitnn.cpu.stats`: one row per operator and shape |

`-Djitnn.verbose=true` also logs the libraries found and each OpenCL placement. All options are in
[CONFIGURATION.md](CONFIGURATION.md).

## Limits

- Storage and accumulation are FP32. The tests check the kernels against the reference backend.
- Only FP32 tensors use the kernels. FP16, BF16, FP8 and quantised tensors run the reference
  implementation, or the OpenCL device.
- Operators with neither a kernel nor an OpenCL device run the reference implementation, which is
  scalar and single-threaded. Check `-Djitnn.cpu.stats=true` when a model is slow.
- Convolutions work in the NCHW layout.

Benchmark results are in [jitNN-benchmarking](https://github.com/beestream-ai/jitNN-benchmarking).
