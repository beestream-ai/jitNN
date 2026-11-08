# Vendor libraries

jitNN runs every operator on its own TornadoVM kernels. With `-Djitnn.native=auto` (or
`-Djitnn.withNative=true`), some operators
call a vendor library instead: cuBLAS, cuDNN, cuFFT and cuSPARSE on CUDA, MLX on Metal.

The calls are TornadoVM library tasks in the same task graph as the JIT kernels. They use the same
device buffers and stream, so nothing is copied to the host between a library call and the kernels
around it.

```bash
java @$TORNADOVM_HOME/tornado-argfile -Djitnn.native=auto ...         # every library available
java @$TORNADOVM_HOME/tornado-argfile -Djitnn.native=cublas,cudnn ...  # only these
```

## Detection and fallback

- **Probes.** The first time an operator could use a library on a device, jitNN runs a one-task
  probe plan with it and checks the result.
- **Unavailable libraries.** A library is unavailable when its TornadoVM module is missing, its
  native library does not load (`libcublas`, `libcudnn`, ...), the probe fails, or the device
  belongs to another backend. It is reported once, and those operators keep jitNN's kernels:

  ```
  [jitnn] native cudnn: not available on NVIDIA A10 (...); using jitNN kernels
  ```
- **Failures at run time.** A plan whose library calls fail on its first execution is compiled
  again with jitNN's kernels. The functions it called that had not yet run in another plan
  (`cudnn.sdpa`, say) are not used again in that JVM; the library's other functions (`cudnn.conv`)
  still are.
- **Plan cache.** The libraries in use are part of the plan cache key.

`-Djitnn.native.unavailable=cudnn,...` treats libraries as missing, to test the fallback.
`-Djitnn.native.verbose=true` logs the libraries found. `Native.emittedTasks()` counts the library
tasks compiled into plans.

## Requirements

- **TornadoVM** develop 8.0.1-dev or later, built with the CUDA or Metal backend
  (`make jdk22plus BACKEND=cuda`). Its SDK includes the library modules and its `tornado-argfile`
  adds them. On Blackwell GPUs (RTX 50 series) use a TornadoVM that loads the CUDA 13 libraries
  before older ones: cuBLAS 12.0, found first on many systems, has no kernels for them.
- **cuDNN** 9 (`libcudnn.so.9`) on the library path. cuBLAS, cuFFT and cuSPARSE come with the CUDA
  toolkit.
- **MLX** needs TornadoVM's Metal backend, `brew install mlx` (for `mlx.metallib`;
  `-Dtornado.mlx.metallib` names another copy) and an Apple M4-class GPU or newer. Validated with
  MLX 0.32.3. jitNN sets `-Dtornado.mlx.async=true` whenever MLX is in use, so MLX calls are
  committed without waiting for the GPU.

## Operators

| Operator | CUDA | Metal (MLX) | Conditions |
|---|---|---|---|
| Matrix products: `matmul`, `linear`, every transpose, the A^T B weight gradients | cuBLAS `Sgemm` | `matmul`, `matmulTransposed` | MLX: A not transposed, arrays of exactly the product's size |
| Batched products, broadcast operands | cuBLAS `SgemmStridedBatched` | `einsumBatchedMatmul` | MLX: no transposes, nothing broadcast |
| FP16 products (FP32 result) | cuBLAS `GemmEx` FP16 → FP32 | – | both operands FP16 |
| `linear` with an FP16 weight (autocast) | FP16 cast + cuBLAS `GemmEx` | – | |
| BF16 products (autocast to BF16, FP32 result) | cuBLAS `GemmEx` BF16 → FP32 | – | a TornadoVM whose cuBLAS binding takes BF16 bits in a `ShortArray` |
| Convolution forward | cuDNN `conv2d`, channels-last (NHWC) in inference | `conv2d` (channels-last) | one group, no dilation, at least `jitnn.native.convMinFlops` multiply-adds (FP32) |
| FP16 convolution forward in `Autocast.enableHalf` inference | cuDNN FP16 `conv2d` on the tensor cores, channels-last | `conv2d` FP16 (channels-last) | one group, no dilation |
| Convolution gradients | cuDNN `ConvolutionBackwardData` and `ConvolutionBackwardFilter` | – | FP32 NCHW, one group, no dilation, at least `jitnn.native.convMinFlops` multiply-adds; a TornadoVM whose cuDNN binding has backward convolutions |
| Other convolution gradients, 1x1 convolutions, Winograd | cuBLAS (the GEMMs jitNN expresses them as) | – | |
| Bias and activation after any of the above | jitNN kernel | jitNN kernel | |
| `softmax` (not the log form) | cuDNN `softmax` | `softmaxRows` | |
| `layerNorm`, `rmsNorm` in inference | – | `layerNorm`, `rmsNorm` | statistics not needed (`Grad.inference`), LayerNorm with a bias |
| FP16 linear layers in `Autocast.enableHalf` inference | cuBLASLt with the bias and ReLU or GELU in its epilogue; cuBLAS `GemmEx` without a bias | – | cuBLASLt's GELU is the tanh approximation; it also stands in for the exact GELU unless `-Djitnn.native.approximateGelu=false` |
| FP16 attention in `Autocast.enableHalf` inference | cuDNN fused attention where it builds a plan on the device (probed once), reading q, k and v in place from a packed projection when TornadoVM's binding has that form; else CUTLASS batched FP16 GEMMs (S = q k^T, O = P v) around jitNN's softmax | – | no mask tensor, log-sum-exp not needed. cuDNN: head dimension a multiple of 8, at most 256. CUTLASS: key length and head dimension multiples of 4 |
| Attention in inference | cuDNN fused SDPA (FP16) with `-Djitnn.native.attentionFp16=true`; falls back where cuDNN builds no plan for the device | `scaledDotProductAttention` | no mask tensor, [B, H, T, D] layout, log-sum-exp not needed. MLX: head dimension 64/72/80/96/128 (decode: 64/96/128/256). cuDNN: head dimension a multiple of 8, at most 256. Causal only with T = S |
| FFT: complex to complex, real to complex | cuFFT | – | full length, contiguous rows |
| Sparse × dense (`SparseMatrix.mm`) | cuSPARSE `SpMM` | – | |

Everything else runs on jitNN's kernels, mostly because TornadoVM has no binding for it:

| Not covered | Reason |
|---|---|
| Grouped and dilated convolutions | the cuDNN binding has one-group, undilated convolutions only |
| BatchNorm, LayerNorm with gradients, pooling, RNN/LSTM/GRU | no binding. Max pooling has no padding argument and no argmax for the gradient |
| Random numbers (cuRAND) | jitNN's Philox streams must match the host and regenerate dropout masks in the backward pass |
| Linear algebra (LU, QR, SVD, ...), SDDMM | no cuSOLVER binding, no SDDMM in cuSPARSE's binding |
| cuBLASLt epilogues for FP32 products, CUTLASS epilogues | they write FP16; FP32 products keep jitNN's bias and activation kernel |

## Options

| Option | Default | Effect |
|---|---|---|
| `jitnn.native.convMinFlops` | 2^30 | smallest FP32 convolution (multiply-adds) cuDNN takes; smaller ones keep jitNN's kernels |
| `jitnn.native.cudnnChannelsLast` | `all` | in inference, keep image tensors [N, H, W, C] between the convolutions cuDNN runs channels-last, converting only around operators that need [N, C, H, W]; `half` limits it to FP16 convolutions, `off` turns it off. Needs a TornadoVM whose cuDNN binding has NHWC convolutions |
| `jitnn.native.tf32` | `false` | run cuBLAS FP32 products on TF32 tensor cores; under `auto`, FP32 convolutions go to cuDNN only with this set (or with `cudnn` named in `jitnn.native`) |
| `jitnn.native.attentionFp16` | `false` | cuDNN fused SDPA in FP32 inference (rounds q, k, v to FP16) |
| `jitnn.native.approximateGelu` | `true` | let cuBLASLt's tanh GELU stand in for the exact GELU |

## Numerics

- **cuBLAS.** FP32 products are FP32 unless `jitnn.native.tf32=true`.
- **cuDNN convolutions.** Under `jitnn.native=auto`, FP32 convolutions go to cuDNN only with
  `jitnn.native.tf32=true`. They use cuDNN's default math, TF32 on Ampere and newer GPUs, for the
  forward convolution and, where available, both gradients. Results differ from FP32 by about 1e-3
  relative; `NVIDIA_TF32_OVERRIDE=0` in the environment makes them FP32.
- **FP16 attention** rounds q, k and v to FP16, with an error of about 5e-3.

## CUDA graphs

Plans that call the libraries are captured as CUDA graphs like any other plan. With cuDNN this needs
TornadoVM develop 7.3.2-dev or newer: older versions free the cuDNN workspace while graphs captured
earlier still use it.

Results with and without the libraries are in
[jitNN-benchmarking](https://github.com/beestream-ai/jitNN-benchmarking).
