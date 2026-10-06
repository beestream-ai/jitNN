# Performance

The KernelContext baseline, measured 2026-10-01. Kernel times come from the TornadoVM profiler
(median of 30 replays); wall times include dispatch. The Apple OpenCL profiler's kernel times are
not usable, so OpenCL rows use wall time.

## Kernels

| Kernel | RTX 5070 Ti, CUDA | M4 Pro, Metal | M4 Pro, MPS (PyTorch) |
|---|---|---|---|
| add, 16M elements | 790 GB/s | 235 GB/s | 239 GB/s |
| exp, 16M | 787 GB/s | 225 GB/s | |
| sum, 16M (two-pass split) | 744 GB/s | 203 GB/s | |
| sum axis 0 / axis 1, 4096x4096 | 737 / 754 GB/s | 123 / 137 GB/s | |
| softmax, 4096x4096 | 753 GB/s | 147 GB/s | 183 GB/s |
| transpose, 4096x4096 | 383 GB/s | ~100 GB/s | |
| SGEMM 1024 | 8.2 TFLOP/s | 2.4 TFLOP/s | 3.8 TFLOP/s |
| SGEMM 2048 | 12.7 TFLOP/s | 2.3 TFLOP/s | 5.4 TFLOP/s |

Peaks for reference: RTX 5070 Ti about 896 GB/s; M4 Pro about 273 GB/s.

## MNIST MLP training step

784-256-128-10, batch 128, Adam, 936 steps.

| | CUDA | Metal | OpenCL |
|---|---|---|---|
| Samples/s | 210k | 28.7k | 32.3k |
| Step | 0.61 ms | 4.47 ms | 3.96 ms |
| jitNN host (executor + tracing) | ~0.2 ms | ~0.1 ms | ~0.2 ms |
| Device kernels in the step | 0.29 ms | 0.49 ms | |

On Apple GPUs most of a small model's step is TornadoVM's per-kernel dispatch, about 0.15 ms per
kernel. Fusion took the step from 31 kernels to 22: bias and ReLU in the GEMM epilogue, and a
fused cross-entropy with a one-kernel backward. On CUDA, CUDA-graph replay took the step from
0.73 to 0.61 ms.

## What the performance pass changed

| Change | Effect |
|---|---|
| Register-blocked GEMM, 64x64 and 128x128 | CUDA 3.9 to 12.7 TFLOP/s; Metal 1.0 to 2.2 TFLOP/s |
| Metal simdgroup GEMM | Metal 2.3 TFLOP/s, ahead of the scalar kernels at 1024 |
| Two-pass split reductions | full sum 3.5 to 72-84 GB/s on M4 Pro |
| Softmax row cached in local memory | Metal 97 to 147 GB/s |
| Fused linear and cross-entropy | 31 to 22 kernels per MLP step; Metal and OpenCL steps 22-34% faster |
| CUDA graphs | CUDA MLP step 16% faster |

## Convolution kernels (M4 Pro, Metal; batch 64)

| Kernel | Forward | Input gradient | Weight gradient | PyTorch MPS forward |
|---|---|---|---|---|
| conv 3x3, 64 channels, 28x28 | 1.9 TFLOP/s | 1.8 TFLOP/s | 1.4 TFLOP/s | 6.1 TFLOP/s |
| conv 3x3, 128 channels, 14x14 | 1.6 | 1.6 | 1.6 | 6.1 |
| conv 3x3, 256 channels, 7x7 | 1.6 | 1.8 | 1.9 | 5.0 |
| conv 1x1, 96 to 576 channels, 7x7 | 1.7 | 1.2 | 1.3 | |
| depthwise 3x3, 576 channels, 7x7 | 0.32 | 0.30 | 0.09 | |

BatchNorm training runs at 196 GB/s and max pooling at 240 GB/s (kernel time).

## CNN training steps (MNIST, batch 64, Adam)

| | Metal | OpenCL (M4 Pro) | CUDA (RTX 5070 Ti) | CUDA (A10) | PyTorch MPS |
|---|---|---|---|---|---|
| ResNet-18 (11.2M parameters, 268 kernels) | 165 ms | 162 ms | 160 ms | 65.6 ms | 54 ms |
| MobileNetV2 (2.2M parameters, 515 kernels) | 148 ms | 138 ms | | 46.7 ms | 62 ms |

The RTX 5070 Ti figure was measured before the TornadoVM fix for result reads from multi-graph
plans (TornadoVM#1159), which the A10 runs include.

On CIFAR-10 (3x32x32 inputs, batch 64, one epoch), steady-state steps:

| | Metal (M4 Pro) | OpenCL (M4 Pro) | CUDA (A10) |
|---|---|---|---|
| ResNet-18 | 189 ms (339 samples/s) | 192 ms (333 samples/s) | 71.7 ms (892 samples/s) |
| MobileNetV2 | 153 ms (418 samples/s) | 139 ms (462 samples/s) | 39.1 ms (1,639 samples/s) |

## Transformer training steps (batch 16 x 128 tokens, 4 layers of 128, AdamW, clipping)

Steady-state steps of the transformer gates, before and after the transformer performance work
(split-K, fused attention, fewer launches, tensor cores):

| | Metal (M4 Pro) | OpenCL (M4 Pro) | CUDA (A10) |
|---|---|---|---|
| GPT-2 (0.84M parameters) | 95 → 87 ms | 92 → 51 ms | 21.7 → 11.7 ms |
| Llama-style (0.87M parameters) | 93 → 88 ms | 85 → 53 ms | 27.8 → 13.8 ms |
| BERT, masked LM (0.86M parameters) | 107 → 101 ms | 96 → 57 ms | 22.7 → 13.3 ms |
| GPT-2, BF16 autocast | 147 → 90 ms | 96 → 54 ms | 14.1 → 4.7 ms |
| Llama, FP16 autocast with loss scaling | 210 → 99 ms | 119 → 67 ms | 29.9 → 11.9 ms |
| BERT, FP16 autocast with loss scaling | 218 → 104 ms | 127 → 63 ms | 15.2 → 6.4 ms |

The loss curves are unchanged: every gate still matches PyTorch to four digits (FP32) or within
0.1% (mixed precision).

What changed, found with the per-operator profile (`TransformerBenchAccelTest` with
`-Djitnn.test.profile=true`, `Engine.kernelBreakdown()`):

| Change | Effect (GPT-2 step) |
|---|---|
| Split-K for weight gradients: A^T B with K = 2048 tokens and a 128-512 output used 4-32 work-groups | those GEMMs 3x faster; device time 82 → 58 ms on Metal |
| Fused attention (FlashAttention-style tiling, online softmax, deterministic backward with recomputation; thread per output element, padded local rows) | scores never stored; attention kernels 30 → 13 ms on Metal |
| Clipping and loss unscaling inside the optimiser update, from one gradient-statistics operator (one reduction per gradient) | 555 → 377 launches |
| Strided attention: the fused kernels read q, k, v from the packed projection (or [B, T, H, d]) and write the merged heads; RoPE on [B, T, H, d] | no permutes, splits or merges around attention; 377 → 317 launches |
| Half-precision operands on Metal's simdgroup matrix units | mixed precision no slower than FP32 on Metal |
| Multi-tensor optimiser step (4 parameters per launch) and gradient statistics (8 gradients per launch) | fewer launches; CUDA GPT-2 12.7 → 11.8 ms |
| Tensor-core GEMMs on CUDA for FP16/BF16 operands (every transpose, split-K) | CUDA GPT-2 BF16 5.8 → 4.7 ms, BERT FP16 7.5 → 6.4 ms |

A step of these small models is still dominated by the number of kernels on Apple GPUs (about
0.15 ms of TornadoVM dispatch each). On the A10, with CUDA graphs, a GPT-2 step spends 7.5
ms in 301 kernels; the GEMMs are about 3 ms of it, the attention kernels 1.4 ms, and the
optimiser and gradient-statistics kernels the rest of the non-GEMM time.

Half-precision GEMMs on the A10 run on tensor cores (`mma.sync` m16n8k16, FP32 accumulation) with
a 128x128 pipelined tile adapted from jitllm, on strided operands so every transpose and the
split-K weight gradients use it. Products whose M and N are not multiples of 128 (or K not a
multiple of 16) fall back to the generic tiles. TFLOP/s, `BaselineBenchAccelTest`,
`-Djitnn.tensorCores=false` vs the default:

| A10, square | NN FP16 | NN BF16 | A B^T FP16 | A B^T BF16 |
|---|---|---|---|---|
| 1024 | 4.2 → 12.7 | 4.2 → 11.3 | 3.7 → 14.2 | 3.8 → 14.5 |
| 2048 | 5.0 → 30.1 | 5.0 → 29.4 | 4.4 → 33.1 | 4.5 → 33.3 |
| 4096 | 5.0 → 30.1 | 5.1 → 30.6 | 4.4 → 33.1 | 4.4 → 33.8 |

Device memory for intermediates, with and without buffer reuse:

| Plan | Without reuse | With reuse |
|---|---|---|
| ResNet-18 training step | 819 MB | 463 MB |
| MobileNetV2 training step | 1,965 MB | 1,091 MB |
| MobileNetV2 evaluation, batch 250 (`Grad.inference`) | 3,810 MB | 427 MB |

## Generative and recurrent training steps (A100)

Steady-state steps of the generative and recurrent gates, PyTorch 2.14 with cuDNN (TF32 off) against jitNN, both
reading the loss back every step:

| Gate | PyTorch (CUDA) | jitNN, CUDA | jitNN, OpenCL |
|---|---|---|---|
| VAE, MNIST, batch 128 | 3.4 ms | 2.2 ms | 5.3 ms |
| DCGAN, MNIST, batch 64 (one D and one G step) | 7.2 ms | 7.5 ms | 14.0 ms |
| U-Net, MNIST, batch 32 | 5.7 ms | 7.8 ms | 12.9 ms |
| LSTM LM, 2 layers of 256, 32 x 64 tokens | 6.6 ms | 36.7 ms | 50.0 ms |
| GRU LM, 2 layers of 256, 32 x 64 tokens | 6.3 ms | 28.4 ms | 40.2 ms |

The recurrent layers are the gap: each timestep is one kernel (one thread per hidden unit, its
gate dot products as plain loops), so a step launches 4 T kernels for two layers and reads W_hh
from memory every timestep, where cuDNN keeps it on chip in a persistent kernel.

On the A100, tensor-core GEMMs reach 55 TFLOP/s (FP16 and BF16, 4096, A B) and 66 TFLOP/s (A B^T);
GPT-2 under BF16 autocast trains at 4.6 ms per step.

## Inference: ResNet-50 and BERT-base (A100)

Latency per call on an NVIDIA A100 80GB, CUDA backend: host input in, host output out, 10 warm-up
calls, then the median of 50 timed calls. FP32 runs have TF32 off in every framework, so they
compare FP32 arithmetic. BERT-base is the encoder (12 layers of 768, hidden states out).

### jitNN's own models

`ResNet.resnet50` and `Bert` built in Java from the same seeds as the Python side, so every
framework runs the same weights on the same inputs (`InferenceBenchAccelTest` and
`scripts/bench/t1.py`), against PyTorch 2.14 and ONNX Runtime 1.30, in ms:

| Model, batch | jitNN | PyTorch | ONNX Runtime |
|---|---|---|---|
| ResNet-50, b1, FP32 | 4.0 | 7.8 | 3.6 |
| BERT-base, b1, seq 128, FP32 | 6.5 | 5.5 | 3.6 |
| BERT-base, b1, seq 128, FP16 | 4.2 | 6.3 | 2.8 |

These include the inference fusions (convolution + BatchNorm folding, GEMM epilogues) and
split-K for forward products. At large batch jitNN is 2-3x behind, set by the speed of its
convolution and GEMM kernels.

### The same models from ONNX files

ResNet-50 and BERT-base exported to ONNX by PyTorch, run by `jitnn-onnx` (`OnnxT1AccelTest`) and
by ONNX Runtime's CUDA provider (`scripts/onnx/t1_onnx.py`) on the same files and inputs, in ms:

| Model | Batch | Precision | jitnn-onnx | ONNX Runtime |
|---|---|---|---|---|
| ResNet-50 | 1 | FP32 | 3.15 | 3.58 |
| ResNet-50 | 32 | FP32 | 46.0 | 20.4 |
| ResNet-50 | 128 | FP32 | 172.9 | 79.2 |
| ResNet-50 | 1 | FP16 | 3.41 | 1.91 |
| ResNet-50 | 32 | FP16 | 46.5 | 5.8 |
| ResNet-50 | 128 | FP16 | 173.2 | 20.6 |
| BERT-base | 1 x 128 | FP32 | 7.48 | 3.62 |
| BERT-base | 32 x 128 | FP32 | 98.2 | 53.3 |
| BERT-base | 8 x 512 | FP32 | 106.5 | 60.3 |
| BERT-base | 1 x 128 | FP16 | 6.25 | 2.45 |
| BERT-base | 32 x 128 | FP16 | 40.2 | 10.4 |
| BERT-base | 8 x 512 | FP16 | 49.0 | 12.0 |

jitNN's outputs agree with ONNX Runtime's to 1.4e-6 (FP32) and 3.3e-3 (FP16), relative. On the
OpenCL backend of the same A100, ResNet-50 at batch 1 takes 9.6 ms and BERT-base at batch 1 (FP32)
20.7 ms.

The remaining gap is kernel speed: jitNN's GEMM, convolution and attention kernels against
cuBLAS and cuDNN. jitNN has no FP16 convolutions yet, so the FP16 ResNet-50 runs its
convolutions in FP32, which explains why it gains nothing from half precision.

To reproduce: `python scripts/bench/t1.py <dir>` exports the models and times PyTorch and ONNX
Runtime; `python scripts/onnx/t1_onnx.py <dir>` writes ONNX Runtime's outputs and timings next to
the models; then
`./mvnw test -Paccel-tests -pl jitnn-onnx -Dtest=OnnxT1AccelTest -Djitnn.onnx.t1=<dir>` runs
jitNN on them. `scripts/bench/djl` holds the same comparison through DJL, as a baseline.

## Known gaps

- Convolutions run on the same register-blocked tiles as the GEMM and share its ceiling: about
  30% of MPS on Metal.
- A CNN step is dominated on Apple GPUs by TornadoVM's per-kernel dispatch cost.
- Metal GEMM is about 42% of MPS. Every tile shape, and the simdgroup path, plateaus at 2.2-2.3
  TFLOP/s; getting further needs vector loads, double buffering and larger SIMD-group tiles.
- Metal transpose and strided reductions sit at 100-140 GB/s.
- LSTM and GRU steps are 4-6x slower than cuDNN (one kernel per timestep; see above).
- Large-batch inference is 2-3x behind ONNX Runtime in FP32 and further in FP16 (no FP16
  convolutions; GEMM and attention kernels slower than cuBLAS and cuDNN).
- Data-parallel ranks run one after another, so several devices give no speed-up yet.
- The Fourier transforms are direct O(n^2) DFTs, and the matrix decompositions use one
  work-group per matrix: fine for the sizes inside models, slow for large inputs.
