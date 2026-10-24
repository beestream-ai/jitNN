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

- **TornadoVM:** develop, 7.3.2-dev or later, built with the CUDA or Metal backend
  (`make jdk22plus BACKEND=cuda`). Its SDK includes the library modules, and its
  `tornado-argfile` adds them. On Blackwell GPUs (RTX 50 series) TornadoVM must load the CUDA 13
  libraries before older ones (beehive-lab/TornadoVM#1280): cuBLAS 12.0, found first on many
  systems, has no kernels for them and runs 3-6x slower.
- **cuDNN:** 9 (`libcudnn.so.9`) on the library path. cuBLAS, cuFFT and cuSPARSE come with the
  CUDA toolkit.
- **MLX:**
  - TornadoVM's Metal backend.
  - `brew install mlx`, for MLX's `mlx.metallib`; `-Dtornado.mlx.metallib` names another copy.
  - An Apple M4-class GPU or newer. Validated on an M4 Pro with MLX 0.32.3.
  - jitNN sets `-Dtornado.mlx.async=true` whenever MLX is in use, so MLX calls are committed without
    waiting for the GPU.

## Operators

| Operator | CUDA | Metal (MLX) | Conditions |
|---|---|---|---|
| Matrix products: `matmul`, `linear`, every transpose, the A^T B weight gradients | cuBLAS `Sgemm` | `matmul`, `matmulTransposed` | MLX: A not transposed, arrays of exactly the product's size |
| Batched products, broadcast operands | cuBLAS `SgemmStridedBatched` | `einsumBatchedMatmul` | MLX: no transposes, nothing broadcast |
| FP16 products (FP32 result) | cuBLAS `GemmEx` FP16 → FP32 | – | both operands FP16 |
| `linear` with an FP16 weight (autocast) | FP16 cast + cuBLAS `GemmEx` | – | |
| Convolution forward | cuDNN `conv2d`, channels-last (NHWC) in inference | `conv2d` (channels-last) | one group, no dilation, at least `jitnn.native.convMinFlops` multiply-adds (FP32) |
| FP16 convolution forward in `Autocast.enableHalf` inference | cuDNN FP16 `conv2d` on the tensor cores, channels-last (NHWC) | `conv2d` FP16 (channels-last) | one group, no dilation |
| Convolution gradients, 1x1 convolutions, Winograd | cuBLAS (the GEMMs jitNN expresses them as) | – | |
| Bias and activation after any of the above | jitNN kernel | jitNN kernel | |
| `softmax` (not the log form) | cuDNN `softmax` | `softmaxRows` | |
| `layerNorm`, `rmsNorm` in inference | – | `layerNorm`, `rmsNorm` | statistics not needed (`Grad.inference`), LayerNorm with a bias |
| FP16 attention in `Autocast.enableHalf` inference | cuDNN's fused attention where it builds a plan on the device (probed once), else CUTLASS batched FP16 GEMMs (S = q k^T, O = P v) around jitNN's softmax | – | no mask tensor, log-sum-exp not needed. cuDNN: head dimension a multiple of 8, at most 256. CUTLASS: key length and head dimension multiples of 4 |
| FP16 linear layers in `Autocast.enableHalf` inference | cuBLASLt with the bias and ReLU or GELU in its epilogue; cuBLAS `GemmEx` without a bias | – | cuBLASLt's GELU is the tanh approximation; it also stands in for the exact GELU unless `-Djitnn.native.approximateGelu=false` |
| Attention in inference | cuDNN fused SDPA (FP16), with `-Djitnn.native.attentionFp16=true`; on the A10 with cuDNN 9.27 it builds no plan, and falls back | `scaledDotProductAttention` | no mask tensor, [B, H, T, D] layout, log-sum-exp not needed. MLX: head dimension 64/72/80/96/128 (decode: 64/96/128/256). cuDNN: head dimension a multiple of 8, at most 256. Causal only with T = S |
| FFT: complex to complex, real to complex | cuFFT | – | full length, contiguous rows |
| Sparse × dense (`SparseMatrix.mm`) | cuSPARSE `SpMM` | – | |

Everything else runs on jitNN's kernels, mostly because TornadoVM has no binding for it:

| Not covered | Reason |
|---|---|
| Convolution backward in cuDNN, grouped and dilated convolutions | the cuDNN binding has forward, one-group convolution only |
| BatchNorm, LayerNorm with gradients, pooling, RNN/LSTM/GRU | no binding for them. Max pooling has no padding argument and no argmax for the gradient |
| Random numbers (cuRAND) | jitNN's Philox streams must match the host and regenerate dropout masks in the backward pass |
| Linear algebra (LU, QR, SVD, ...), SDDMM | no cuSOLVER binding, no SDDMM in cuSPARSE's binding |
| BF16 products | jitNN holds BF16 in `ShortArray`; the bindings take `BFloat16Array` |
| cuBLASLt epilogues for FP32 products, CUTLASS epilogues | they write FP16; FP32 products keep jitNN's bias and activation kernel |

## Numerics

- **cuBLAS.** FP32 products are FP32. `-Djitnn.native.tf32=true` runs them on TF32 tensor cores.
- **cuDNN convolutions** use cuDNN's default math, which is TF32 on Ampere and newer GPUs, as in
  PyTorch by default.
  - Results differ from FP32 by about 1e-3 relative.
  - `NVIDIA_TF32_OVERRIDE=0` in the environment makes them FP32.
- **FP16 attention** rounds q, k and v to FP16, with an error of about 5e-3.

## CUDA graphs

Plans that call the libraries are captured as CUDA graphs like any other plan. With cuDNN this needs
a TornadoVM that includes beehive-lab/TornadoVM#1238 (`develop` 7.3.2-dev or newer): older versions free the
cuDNN workspace while graphs captured earlier still use it.

cuDNN only takes FP32 convolutions of at least `jitnn.native.convMinFlops` multiply-adds (default
2^30). Smaller ones keep jitNN's kernels.

In inference the channels-last pass keeps image tensors [N, H, W, C] between the convolutions cuDNN
runs channels-last (NHWC), its tensor cores' layout, and converts only around the operators that need
[N, C, H, W]. On an RTX 5070 Ti this makes ResNet-50 16% (FP32) to 22% (FP16, batch 128) faster than
cuDNN's channels-first (NCHW) convolutions. `-Djitnn.native.cudnnChannelsLast=half` keeps it to FP16
convolutions, `off` turns it off. It needs a TornadoVM whose cuDNN binding has the NHWC convolutions.

## Results

jitNN's own kernels against vendor libraries, median ms (2026-10-23 on NVIDIA, 2026-10-18 on the
M4 Pro). The same models against PyTorch and ONNX Runtime are in
[PERFORMANCE.md](PERFORMANCE.md#inference-against-pytorch-and-onnx-runtime).

| Workload | A100 kernels | A100 libraries | RTX 5070 Ti kernels | RTX 5070 Ti libraries | M4 Pro kernels | M4 Pro MLX |
|---|---|---|---|---|---|---|
| ResNet-50 b1 FP32 | 3.46 | 3.40 | 1.99 | 1.98 | 7.99 | 7.80 |
| ResNet-50 b32 FP32 | 24.1 | 11.3 | 24.2 | 14.8 | 115.4 | 82.7 |
| ResNet-50 b128 FP32 | 86.9 | 37.8 | 95.7 | 58.8 | 455.7 | 320.9 |
| ResNet-50 b1 FP16 | 2.92 | 1.60 | 1.94 | 0.88 | 7.76 | 7.51 |
| ResNet-50 b32 FP16 | 24.3 | 7.81 | 24.7 | 9.06 | 115.6 | 66.3 |
| ResNet-50 b128 FP16 | 87.9 | 26.3 | 97.8 | 35.3 | 460.0 | 258.9 |
| BERT-base b1 x 128 FP32 | 4.95 | 3.90 | 4.50 | 3.00 | 21.3 | 8.94 |
| BERT-base b32 x 128 FP32 | 62.9 | 50.2 | 76.1 | 41.4 | 388.4 | 164.4 |
| BERT-base b8 x 512 FP32 | 69.2 | 55.4 | 82.9 | 47.4 | 508.4 | 176.5 |
| BERT-base b1 x 128 FP16 | 3.50 | 1.49 | 2.16 | 0.95 | 16.8 | 7.43 |
| BERT-base b32 x 128 FP16 | 20.1 | 7.27 | 20.5 | 11.9 | 351.5 | 144.6 |
| BERT-base b8 x 512 FP16 | 26.0 | 8.19 | 27.2 | 12.6 | 473.5 | 176.2 |

DJL's examples on an A10, unchanged:

| Example | jitNN kernels | `--with-native` |
|---|---|---|
| `TrainResnetWithCifar10`, 100 batches of 32 (whole run) | 60.3 s | 39.4 s |
| `TrainMnist`, 1 epoch (whole run) | 7.3 s | 6.2 s |
| `YoloDetection` (yolo11n), per prediction | 24.9 ms | 26.8 ms |

Validation accuracy matches with and without the libraries:
- DJL ResNet-50: 0.324 and 0.321.
- DJL MNIST: 0.944 and 0.961.
