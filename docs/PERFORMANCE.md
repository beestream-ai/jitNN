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

## Known gaps

- Metal GEMM is about 42% of MPS. Every tile shape, and the simdgroup path, plateaus at 2.2-2.3
  TFLOP/s; getting further needs vector loads, double buffering and larger SIMD-group tiles.
- Metal transpose and strided reductions sit at 100-140 GB/s.
- The optimiser still launches one kernel per parameter. A fused multi-tensor step needs storage
  views into one flat buffer.
