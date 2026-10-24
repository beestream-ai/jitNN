# Performance

[Inference against PyTorch and ONNX Runtime](#inference-against-pytorch-and-onnx-runtime) comes
first; the sections after it cover jitNN's own kernels, training, fusion, autotuning, several GPUs
and compile time.

## Inference against PyTorch and ONNX Runtime

ResNet-50 and BERT-base (the encoder, 12 layers of 768), built by jitNN from the same seeds as the
Python side, so every framework runs the same weights on the same inputs (`InferenceBenchAccelTest`
and `scripts/bench/t1.py`). Each timed call is host input in, forward pass, host output out; 10
warm-up calls, then the median of 50. FP32 runs have TF32 off in every framework. FP16 rows run the
whole model in half precision in every framework (`Autocast.enableHalf` in jitNN). jitNN's FP32
outputs match PyTorch's to 3e-7 to 1e-6, its FP16 outputs to about 4e-4. Every run waited for an
idle GPU below 45 C, with clocks and throttle reasons sampled: no thermal throttling.

Columns: jitNN's own kernels, jitNN with run-time autotuning (`-Djitnn.autotune`), jitNN with vendor
libraries (`-Djitnn.native=auto`, see [NATIVE.md](NATIVE.md)), PyTorch 2.14, ONNX Runtime (CUDA
provider). Fusion is on. Fastest in bold.

### NVIDIA A100 80GB PCIe (CUDA 13.4, 2026-10-23)

| Workload | jitNN kernels | autotuned | vendor libraries | PyTorch | ONNX Runtime 1.30 |
|---|---|---|---|---|---|
| ResNet-50 b1 FP32 | 3.46 | 3.56 | **3.40** | 8.12 | 4.05 |
| ResNet-50 b32 FP32 | 24.1 | 23.8 | **11.3** | 31.7 | 29.3 |
| ResNet-50 b128 FP32 | 86.9 | 87.0 | **37.8** | 87.1 | 78.7 |
| ResNet-50 b1 FP16 | 2.92 | 2.89 | **1.60** | 9.78 | 3.57 |
| ResNet-50 b32 FP16 | 24.3 | 23.9 | **7.81** | 14.6 | 8.16 |
| ResNet-50 b128 FP16 | 87.9 | 87.7 | 26.3 | 28.3 | **20.5** |
| BERT-base b1 x 128 FP32 | 4.95 | 4.89 | 3.90 | 5.56 | **3.59** |
| BERT-base b32 x 128 FP32 | 62.9 | 62.9 | 50.2 | **45.7** | 53.6 |
| BERT-base b8 x 512 FP32 | 69.2 | 69.1 | 55.4 | **49.2** | 60.4 |
| BERT-base b1 x 128 FP16 | 3.50 | 3.45 | **1.49** | 6.13 | 2.59 |
| BERT-base b32 x 128 FP16 | 20.1 | 18.8 | 7.27 | **6.24** | 10.4 |
| BERT-base b8 x 512 FP16 | 26.0 | 24.2 | 8.19 | **6.70** | 12.0 |

### NVIDIA RTX 5070 Ti (CUDA 13.0, 2026-10-23)

| Workload | jitNN kernels | autotuned | vendor libraries | PyTorch | ONNX Runtime 1.31 |
|---|---|---|---|---|---|
| ResNet-50 b1 FP32 | 1.99 | 1.99 | **1.98** | 3.91 | 3.18 |
| ResNet-50 b32 FP32 | 24.2 | 24.2 | **14.8** | 22.3 | 18.7 |
| ResNet-50 b128 FP32 | 95.7 | 95.0 | **58.8** | 93.9 | 78.7 |
| ResNet-50 b1 FP16 | 1.94 | 1.91 | **0.88** | 1.72 | 0.90 |
| ResNet-50 b32 FP16 | 24.7 | 24.3 | 9.06 | 11.4 | **7.6** |
| ResNet-50 b128 FP16 | 97.8 | 95.3 | 35.3 | 50.5 | **34.2** |
| BERT-base b1 x 128 FP32 | 4.50 | 3.99 | 3.00 | 2.95 | **2.40** |
| BERT-base b32 x 128 FP32 | 76.1 | 49.0 | 41.4 | **36.8** | 39.5 |
| BERT-base b8 x 512 FP32 | 82.9 | 57.2 | 47.4 | **40.8** | 45.5 |
| BERT-base b1 x 128 FP16 | 2.16 | 2.13 | **0.95** | 1.35 | 1.28 |
| BERT-base b32 x 128 FP16 | 20.5 | 19.4 | 11.9 | **9.8** | 14.1 |
| BERT-base b8 x 512 FP16 | 27.2 | 27.8 | 12.6 | **10.5** | 17.0 |

On this GPU the vendor-library rows need a TornadoVM that loads the CUDA 13 libraries before older
ones (beehive-lab/TornadoVM#1280): cuBLAS 12.0 has no kernels for Blackwell GPUs and ran SGEMM at
9.7 TFLOP/s against 33.6 for cuBLAS 13.1, which made BERT slower with vendor libraries than without.

### Apple M4 Pro (Metal, 2026-10-18)

PyTorch 2.14.1 on MPS; ONNX Runtime 1.31.0 with the CoreML provider on the CPU and GPU (not the
Neural Engine); jitNN with MLX 0.32.3 (`scripts/bench/run_mac.sh`).

| Workload | jitNN kernels | jitNN with MLX | PyTorch MPS | ONNX Runtime CoreML |
|---|---|---|---|---|
| ResNet-50 b1 FP32 | 7.99 | 7.80 | 10.11 | **5.84** |
| ResNet-50 b32 FP32 | 115.4 | 82.7 | 99.3 | **60.9** |
| ResNet-50 b128 FP32 | 455.7 | 320.9 | 383.7 | **248.0** |
| ResNet-50 b1 FP16 | 7.76 | 7.51 | 9.34 | **3.00** |
| ResNet-50 b32 FP16 | 115.6 | 66.3 | 81.6 | **57.1** |
| ResNet-50 b128 FP16 | 460.0 | 258.9 | 308.1 | **227.4** |
| BERT-base b1 x 128 FP32 | 21.3 | 8.94 | 8.11 | **6.64** |
| BERT-base b32 x 128 FP32 | 388.4 | 164.4 | **160.3** | 166.8 |
| BERT-base b8 x 512 FP32 | 508.4 | 176.5 | **172.7** | 173.1 |
| BERT-base b1 x 128 FP16 | 16.8 | 7.43 | 7.47 | **6.57** |
| BERT-base b32 x 128 FP16 | 351.5 | 144.6 | **133.9** | 141.7 |
| BERT-base b8 x 512 FP16 | 473.5 | 176.2 | **144.3** | 149.9 |

The BERT b8 x 512 FP16 row was timed on a busy machine (re-timings spread from 157 to 194 ms).

### Reading

- With vendor libraries jitNN is the fastest on 6 of 12 workloads on the A100 and 5 on the RTX
  5070 Ti, ResNet-50 FP32 1.3-2.6x ahead at every batch. Elsewhere it is at most 1.28x behind the
  faster of PyTorch and ONNX Runtime: FP16 BERT at large batch (attention) and ResNet-50 FP16 at
  batch 128.
- On the M4 Pro, MLX makes ResNet-50 1.2-1.3x faster than PyTorch MPS and BERT-base level with
  it; ONNX Runtime with CoreML compiles the whole network with Apple's kernels and stays ahead.
- On its own kernels jitNN is level with or ahead of PyTorch on ResNet-50 FP32 on NVIDIA GPUs and
  1.3-3x behind on BERT and FP16. Autotuning closes a third of the BERT FP32 gap on the RTX 5070 Ti.

To reproduce: `python scripts/bench/t1.py <dir>` exports the models and times PyTorch and ONNX
Runtime; `python scripts/onnx/t1_onnx.py <dir>` writes ONNX Runtime's outputs and timings next to
the models; `./mvnw test -Paccel-tests -pl jitnn-onnx -Dtest=OnnxT1AccelTest -Djitnn.onnx.t1=<dir>`
runs the same files through `jitnn-onnx`.

### DJL examples

DJL's examples on an A10, run unchanged on the jitNN engine, with jitNN's kernels and with vendor
libraries (`scripts/bench/djl-examples`):

| Example | jitNN kernels | `--with-native` |
|---|---|---|
| `TrainResnetWithCifar10`, 100 batches of 32 (whole run) | 60.3 s | 39.4 s |
| `TrainMnist`, 1 epoch (whole run) | 7.3 s | 6.2 s |
| `YoloDetection` (yolo11n), per prediction | 24.9 ms | 26.8 ms |

DJL's inference examples give the same results on jitNN as on DJL's PyTorch and ONNX Runtime
engines: `ImageClassification` returns identical probabilities, and `YoloDetection` finds the
frisbee at 0.915 and the dog at 0.657 (ONNX Runtime: 0.917 and 0.661).

## Baseline kernels

The KernelContext baseline, measured 2026-10-01. Kernel times come from the TornadoVM profiler
(median of 30 replays); wall times include dispatch. The Apple OpenCL profiler's kernel times are
not usable, so OpenCL rows use wall time.

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

The recurrent layers were the gap in these runs: each timestep was one kernel (one thread per
hidden unit, its gate dot products as plain loops), so a step launched 4 T kernels for two layers
and read W_hh from memory every timestep, where cuDNN keeps it on chip in a persistent kernel.
LSTM layers of up to 512 hidden units and 256 rows now run the whole sequence in one launch, a
work-group per batch row, forward and backward (`-Djitnn.rnn`, see
[CONFIGURATION.md](CONFIGURATION.md#kernel-variants)); the LSTM row above predates that. GRU layers
still launch per timestep.

On the A100, tensor-core GEMMs reach 55 TFLOP/s (FP16 and BF16, 4096, A B) and 66 TFLOP/s (A B^T);
GPT-2 under BF16 autocast trains at 4.6 ms per step.

## Data-parallel training (2x A10)

Two NVIDIA A10s with no peer-to-peer link (NCCL goes through host shared memory), jitNN's kernels
(`-Djitnn.native=off`), SGD with momentum, the same batch on every device, median step of 30
(`DataParallelBenchAccelTest`, one run per mode for ResNet-50).

| Model | Batch per device | 1 device | 2 devices, host collective | 2 devices, NCCL | NCCL scaling |
|---|---|---|---|---|---|
| ResNet-18, 32 x 32, 10 classes | 128 | 67.0 ms, 1910 samples/s | 192.8 ms, 1328 samples/s | 69.7 ms, 3674 samples/s | 1.92x |
| ResNet-50, 224 x 224, 1000 classes | 32 | 213.0 ms, 150 samples/s | 494.3 ms, 129 samples/s | 224.0 ms, 286 samples/s | 1.91x |

With cuBLAS (`-Djitnn.native=cublas`), ResNet-18 scales 1.90-1.95x over six runs. The host
collective is slower than one device: every gradient goes to the host and back. NCCL keeps the
gradients on the GPUs and reduces them in one call per step, inside each rank's plan.
ResNet-50 at batch 32 needs `-XX:MaxDirectMemorySize` above the 32 GB default for two ranks: each
rank's training step is one plan, and TornadoVM keeps a host copy of every device buffer.

## Fusion: kernels and memory traffic

`-Djitnn.fuse=true` (the default) against `false`, RTX 5070 Ti, TornadoVM develop 801ba0098,
jitNN's kernels (`-Djitnn.native=off`), FP32, medians. Kernels and MB are per call, from
`Engine.trafficStats()`: MB counts each operator's input and output tensors once, the least it
can read and write.

| Workload | Kernels | MB | Time |
|---|---|---|---|
| ResNet-50 inference b1 | 200 to 131 (-35%) | 394 to 261 (-34%) | 1.97 to 2.00 ms |
| ResNet-50 inference b32 | 181 to 112 (-38%) | 9435 to 5177 (-45%) | 28.5 to 24.2 ms (-15%) |
| ResNet-50 inference b128 | 181 to 112 (-38%) | 37433 to 20400 (-46%) | 116.8 to 95.1 ms (-19%) |
| BERT-base inference b1 x 128 | 172 to 160 (-7%) | 618 to 580 (-6%) | 4.41 to 4.45 ms |
| BERT-base inference b32 x 128 | 124 to 112 (-10%) | 6240 to 5032 (-19%) | 76.3 to 75.0 ms (-2%) |
| BERT-base inference b8 x 512 | 124 to 112 (-10%) | 6243 to 5035 (-19%) | 84.5 to 83.2 ms (-2%) |
| ResNet-18 training step, b128 | 344 to 327 (-5%) | 5832 to 5387 (-8%) | 31.06 to 31.15 ms |

Nsight Compute's DRAM bytes over the same calls (sum of `dram__bytes_op_read` and
`dram__bytes_op_write` over the call's kernels, CUDA graphs off) confirm the reductions: ResNet-50
b32 8395 to 5483 MB (-35%), BERT b32 4433 to 3460 MB (-22%). They differ from the counts because
the 64 MB L2 cache serves part of the traffic and kernels re-read tiles. ResNet-50 gains most:
BatchNorm folds into the convolutions and the bias, residual add and ReLU into their epilogues.
In BERT the products dominate and fusion only joins their neighbours.

The fused element-wise chains that remain in training (gradient chains, mostly two operations)
run as kernels specialised per chain at compile time (`fusedProgram1` to `6`): an interpreter
kernel over the same chains made the ResNet-18 step 8% slower than not fusing.

## Autotuning

`-Djitnn.autotune=on` once (the tuning pass), then `use`; jitNN's kernels (`-Djitnn.native=off`),
FP32, medians, off and `use` alternated twice (the runs agree within 1%). The A10 also has an
offline GEMM table; the RTX 5070 Ti has none, so its rules were measured on other GPUs.

| Workload | A10 off | A10 autotuned | RTX 5070 Ti off | RTX 5070 Ti autotuned |
|---|---|---|---|---|
| BERT-base b32 x 128 | 127.2 ms | 107.9 ms (-15%) | 73.5 ms | 48.7 ms (-34%) |
| BERT-base b8 x 512 | 140.2 ms | 120.3 ms (-14%) | 79.3 ms | 54.6 ms (-31%) |
| BERT-base b1 x 128 | 7.25 ms | 7.01 ms (-3%) | 4.67 ms | 4.09 ms (-12%) |
| ResNet-50 b32 | 48.9 ms | 48.4 ms (-1%) | 24.8 ms | 24.6 ms (-1%) |
| ResNet-50 b128 | 189.8 ms | 189.0 ms | 96.2 ms | 96.0 ms |
| ResNet-18 training, b128 | 1830 samples/s | 1874 samples/s (+2.4%) | 3810 samples/s | 3870 samples/s (+1.6%) |

The BERT gain is its linear layers (4096 x 768 to 3072 products), which take 128 x 128 tiles
instead of the rule's 64 x 64, and its layer norms. ResNet's convolutions keep the rules' variants.
The tuning pass took 30-40 s for the training step and 70-100 s for the two inference models
(54 decisions per device), once.

## Compile time

The first call of each plan compiles its kernels. TornadoVM `develop` compiles them with its TC
compiler, which took 33.0 s for the 3867 kernels of jitNN's golden tests on an RTX 5070 Ti host,
against 12.5 s with the Graal-based compiler it replaced. beehive-lab/TornadoVM#1281 brings that to
8.1 s with the same generated code; the golden tests then run in 13.3 s instead of 40.2 s.
`-Djitnn.verbose=true` prints each plan's first-execution time, split into compiler and driver time.

## Known gaps


- Convolutions run on the same register-blocked tiles as the GEMM and share its ceiling: about
  30% of MPS on Metal.
- A CNN step is dominated on Apple GPUs by TornadoVM's per-kernel dispatch cost.
- Metal GEMM is about 42% of MPS. Every tile shape, and the simdgroup path, plateaus at 2.2-2.3
  TFLOP/s; getting further needs vector loads, double buffering and larger SIMD-group tiles.
- Metal transpose and strided reductions sit at 100-140 GB/s.
- GRU steps are 4-5x slower than cuDNN (one kernel per timestep; see above). LSTM layers now run
  fused, but have not been measured against cuDNN again.
- On its own kernels, BERT and FP16 inference are 1.3-3x behind PyTorch (GEMM, convolution and
  attention kernels slower than cuBLAS and cuDNN). With vendor libraries the largest gaps are FP16
  BERT at large batch (1.2x behind PyTorch, attention) and ResNet-50 FP16 at batch 128 (1.28x
  behind ONNX Runtime on the A100).
- Cold start: the first call of a large model spends several seconds compiling kernels (about
  9.6 s for YOLO11n through DJL). Later calls of the same shapes replay the compiled plan, and a
  reloaded model reuses the plans of a closed one.
- With MLX, the bias and activation after MLX's products and convolutions run as separate jitNN
  kernels: TornadoVM's MLX binding has no epilogues.
- cuDNN convolutions use TF32 on Ampere and newer GPUs, as PyTorch does by default; set
  `NVIDIA_TF32_OVERRIDE=0` for FP32 arithmetic.
- The gradient all-reduce of data-parallel training does not overlap the backward pass: a rank's
  plan runs on one TornadoVM stream.
- The Fourier transforms are direct O(n^2) DFTs, and the matrix decompositions use one
  work-group per matrix: fine for the sizes inside models, slow for large inputs.
