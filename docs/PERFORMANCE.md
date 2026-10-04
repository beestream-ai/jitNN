# Performance

Wave 1, KernelContext baseline, measured 2026-10-01. Kernel times come from the TornadoVM profiler
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

## Wave 2 kernels (M4 Pro, Metal; batch 64)

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

| | Metal (M4 Pro) | OpenCL (M4 Pro) | CUDA (A10) |
|---|---|---|---|
| GPT-2 (0.84M parameters) | 95 ms (21.6k tokens/s) | 92 ms (22.2k tokens/s) | 21.7 ms (94k tokens/s) |
| Llama-style (0.87M parameters) | 93 ms (22.0k tokens/s) | 85 ms (24.1k tokens/s) | 27.8 ms (74k tokens/s) |
| BERT, masked LM (0.86M parameters) | 107 ms (19.2k tokens/s) | 96 ms (21.3k tokens/s) | 22.7 ms (90k tokens/s) |

With autocast, the A10 runs GPT-2 (BF16) in 14.1 ms and BERT (FP16 with loss scaling) in
15.2 ms, about a third faster than FP32, and Llama in 29.9 ms (27.5 ms in FP32). On the M4 Pro the
same steps are slower than FP32 (96-127 ms on OpenCL, 147-173 ms on Metal): each product gains
cast kernels for its operands and gradients, kernel dispatch dominates on Apple GPUs, and the
half-precision GEMMs convert as they load, without matrix units yet. Running them on simdgroup
matrices and tensor cores is a later optimisation.

These models are small, so a step is dominated by kernel dispatch (several hundred kernels) rather
than arithmetic. Attention runs as two batched GEMMs around a fused scaled, causal softmax; a fused
attention kernel is the next step.

Device memory for intermediates, with and without buffer reuse:

| Plan | Without reuse | With reuse |
|---|---|---|
| ResNet-18 training step | 819 MB | 463 MB |
| MobileNetV2 training step | 1,965 MB | 1,091 MB |
| MobileNetV2 evaluation, batch 250 (`Grad.inference`) | 3,810 MB | 427 MB |

## Known gaps

- Convolutions run on the same register-blocked tiles as the GEMM and share its ceiling: about
  30% of MPS on Metal.
- A CNN step is dominated on Apple GPUs by TornadoVM's per-kernel dispatch cost.
- Metal GEMM is about 42% of MPS. Every tile shape, and the simdgroup path, plateaus at 2.2-2.3
  TFLOP/s; getting further needs vector loads, double buffering and larger SIMD-group tiles.
- Metal transpose and strided reductions sit at 100-140 GB/s.
- The optimiser still launches one kernel per parameter. A fused multi-tensor step needs storage
  views into one flat buffer.
