# Vendor libraries (`--with-native`)

jitNN runs every operator on its own TornadoVM kernels. With `-Djitnn.native=auto` (or
`-Djitnn.withNative=true`, which the `--with-native` option of jitNN's tools sets), some operators
call a vendor library instead:

- **CUDA:** cuBLAS, cuDNN, cuFFT and cuSPARSE.
- **Metal:** MLX.

The calls are TornadoVM library tasks inside the same task graph as the JIT kernels. They work on
the same device buffers and stream, so nothing is copied to the host between a library call and
the kernels around it.

```bash
java @$TORNADOVM_HOME/tornado-argfile -Djitnn.native=auto ...         # every library available
java @$TORNADOVM_HOME/tornado-argfile -Djitnn.native=cublas,cudnn ...  # only these
```

## Detection and fallback

- **Probes.** The first time an operator could use a library on a device, jitNN runs a one-task
  probe plan with that library and checks its result.
- **Unavailable libraries.** A library counts as unavailable when:
  - its TornadoVM module is missing;
  - its native library does not load (`libcublas`, `libcudnn`, ...);
  - the probe fails;
  - the device belongs to another backend.

  An unavailable library is reported once, and those operators keep jitNN's kernels:

  ```
  [jitnn] native cudnn: not available on NVIDIA A10 (...); using jitNN kernels
  ```
- **Failures at run time.** A plan whose library calls fail on its first execution is compiled
  again with jitNN's kernels.
  - The functions it called that had not yet run in another plan (`cudnn.sdpa`, say) are not used
    again in that JVM.
  - The library's other functions (`cudnn.conv`) still are.
  - `-Djitnn.native.unavailable=cudnn,...` treats libraries as missing, to test the fallback.
- **Plan cache.** The libraries in use are part of the plan cache key.

`-Djitnn.native.verbose=true` also logs the libraries found. `Native.emittedTasks()` counts the
library tasks compiled into plans.

## Requirements

- **TornadoVM:** develop, 7.2.1-dev or later, built with the CUDA backend
  (`bin/compile --jdk jdk22plus --backend cuda`). Its SDK includes the library modules, and its
  `tornado-argfile` adds them.
- **cuDNN:** 9 (`libcudnn.so.9`) on the library path. cuBLAS, cuFFT and cuSPARSE come with the
  CUDA toolkit.
- **MLX:**
  - TornadoVM's Metal backend.
  - `brew install mlx`, for MLX's `mlx.metallib`; `-Dtornado.mlx.metallib` names another copy.
  - An Apple M4-class GPU or newer.

## Operators

| Operator | CUDA | Metal (MLX) | Conditions |
|---|---|---|---|
| Matrix products: `matmul`, `linear`, every transpose, the A^T B weight gradients | cuBLAS `Sgemm` | `matmul`, `matmulTransposed` | MLX: A not transposed, arrays of exactly the product's size |
| Batched products, broadcast operands | cuBLAS `SgemmStridedBatched` | `einsumBatchedMatmul` | MLX: no transposes, nothing broadcast |
| FP16 products (FP32 result) | cuBLAS `GemmEx` FP16 → FP32 | – | both operands FP16 |
| `linear` with an FP16 weight (autocast) | FP16 cast + cuBLAS `GemmEx` | – | |
| Convolution forward | cuDNN `conv2d` (NCHW) | – | one group, no dilation, at least `jitnn.native.convMinFlops` multiply-adds |
| Convolution gradients, 1x1 convolutions, Winograd | cuBLAS (the GEMMs jitNN expresses them as) | – | |
| Bias and activation after any of the above | jitNN kernel | jitNN kernel | |
| `softmax` (not the log form) | cuDNN `softmax` | `softmaxRows` | |
| `layerNorm`, `rmsNorm` in inference | – | `layerNorm`, `rmsNorm` | statistics not needed (`Grad.inference`), LayerNorm with a bias |
| Attention in inference | cuDNN fused SDPA (FP16), with `-Djitnn.native.attentionFp16=true`; on the A10 with cuDNN 9.27 it builds no plan, and falls back | `scaledDotProductAttention` | no mask tensor, [B, H, T, D] layout, log-sum-exp not needed. MLX: head dimension 64/72/80/96/128 (decode: 64/96/128/256). cuDNN: head dimension a multiple of 8, at most 256. Causal only with T = S |
| FFT: complex to complex, real to complex | cuFFT | – | full length, contiguous rows |
| Sparse × dense (`SparseMatrix.mm`) | cuSPARSE `SpMM` | – | |

Everything else runs on jitNN's kernels, mostly because TornadoVM has no binding for it:

| Not covered | Reason |
|---|---|
| Convolution backward in cuDNN, NHWC layouts, grouped and dilated convolutions | the cuDNN binding has forward, NCHW, one-group convolution only |
| BatchNorm, LayerNorm with gradients, pooling, RNN/LSTM/GRU | no binding for them. Max pooling has no padding argument and no argmax for the gradient |
| Random numbers (cuRAND) | jitNN's Philox streams must match the host and regenerate dropout masks in the backward pass |
| Linear algebra (LU, QR, SVD, ...), SDDMM | no cuSOLVER binding, no SDDMM in cuSPARSE's binding |
| BF16 products | jitNN holds BF16 in `ShortArray`; the bindings take `BFloat16Array` |
| cuBLASLt and CUTLASS fused epilogues | FP16 outputs; jitNN's products write FP32 |
| NCCL | jitNN's ranks run one at a time in the executor |

## Numerics

- **cuBLAS.** FP32 products are FP32. `-Djitnn.native.tf32=true` runs them on TF32 tensor cores.
- **cuDNN convolutions** use cuDNN's default math, which is TF32 on Ampere and newer GPUs, as in
  PyTorch by default.
  - Results differ from FP32 by about 1e-3 relative.
  - `NVIDIA_TF32_OVERRIDE=0` in the environment makes them FP32.
- **FP16 attention** rounds q, k and v to FP16, with an error of about 5e-3.

## CUDA graphs

Plans that call cuDNN run without a CUDA graph. The reason is in TornadoVM:
- cuDNN's workspace is shared by all of a plan's task graphs.
- It is reallocated when a later convolution needs more.
- Graphs captured before that point would then use freed memory.

To keep small-batch plans on CUDA graphs, where launches matter more than the convolutions, cuDNN
only takes convolutions of at least `jitnn.native.convMinFlops` multiply-adds (default 2^30).
Smaller ones keep jitNN's kernels.

## Results

A10 (kobol), TornadoVM develop e3381a277, medians.

| Workload | jitNN kernels | `--with-native` | Speedup |
|---|---|---|---|
| ResNet-50 inference, batch 1 | 4.05 ms | 4.04 ms | 1.00x |
| ResNet-50 inference, batch 32 | 52.2 ms | 33.3 ms | 1.57x |
| ResNet-50 inference, batch 128 | 217.9 ms | 126.9 ms | 1.72x |
| BERT-base FP32, batch 1 × 128 | 7.08 ms | 4.11 ms | 1.72x |
| BERT-base FP32, batch 32 × 128 | 125.6 ms | 82.4 ms | 1.52x |
| BERT-base FP32, batch 8 × 512 | 139.2 ms | 96.0 ms | 1.45x |
| BERT-base FP16, batch 1 × 128 | 4.79 ms | 2.95 ms | 1.62x |
| BERT-base FP16, batch 32 × 128 | 48.3 ms | 38.5 ms | 1.25x |
| BERT-base FP16, batch 8 × 512 | 61.1 ms | 51.4 ms | 1.19x |
| DJL `TrainResnetWithCifar10`, 100 batches of 32 (whole run) | 60.3 s | 39.4 s | 1.53x |
| DJL `TrainMnist`, 1 epoch (whole run) | 7.3 s | 6.2 s | 1.18x |
| DJL `YoloDetection` (yolo11n), per prediction | 24.9 ms | 26.8 ms | 0.93x |

Validation accuracy matches with and without the libraries:
- DJL ResNet-50: 0.324 and 0.321.
- DJL MNIST: 0.944 and 0.961.
