::::{only} html
```{image} images/jitnn-logo-light.svg
:alt: jitNN
:width: 480px
:align: center
```
::::

# jitNN

**jitNN** is a machine-learning and deep-learning library for the JVM, written in Java on
[TornadoVM](https://github.com/beehive-lab/TornadoVM). Tensors, autograd, layers and optimisers
are plain Java, and every operator is a TornadoVM kernel that is JIT-compiled for the GPU it runs
on. Its operator set covers what DL4J/ND4J, DJL and Deep Netts provide, and adds linear algebra,
Fourier transforms, sparse and graph operators, classical machine learning and an ONNX importer.

- **Portable.** Every operator has a KernelContext kernel, forward and backward, so the same code
  runs on CUDA, OpenCL and Metal. Faster backend-specific variants (Metal simdgroup matrices,
  tensor-core GEMMs on NVIDIA) sit behind the same operator and are checked against the baseline.
- **Compiled whole.** Operators are recorded lazily. At a sync point the forward pass, the backward
  pass and the optimiser step compile into one TornadoVM task graph, which is cached by structure
  and replayed on every following training step, as a CUDA graph on NVIDIA. Parameters stay on the
  device between steps.
- **Validated against PyTorch, scikit-learn and ONNX Runtime.** Every operator is checked against
  PyTorch golden tensors, forward and gradients, on every device, and every gradient against finite
  differences. Models are trained from the same initial weights as PyTorch: an MNIST MLP tracks its
  loss to 1e-6; ResNet-18 and MobileNetV2 match PyTorch's FP64 gradients as closely as PyTorch's
  own FP32 does and reach its CIFAR-10 accuracy; GPT-2, a Llama-style decoder and BERT end on
  PyTorch's losses to four digits (within 0.1% under BF16 or FP16 autocast). The classical
  estimators match scikit-learn, and imported ONNX models match ONNX Runtime.
- **Fused and frugal.** Bias, ReLU and GELU run in the GEMM and convolution epilogues, BatchNorm is
  folded into convolutions for inference, cross-entropy's backward is a single kernel, and dead
  buffers are reused inside a compiled step (-43% memory on a ResNet-18 training step). On CUDA the
  element-wise, reduction and softmax kernels reach 83-88% of peak memory bandwidth.

## What is in it

| Area | Contents |
|---|---|
| Tensors and autograd | FP32, INT32, FP16, BF16, FP8 (E4M3, E5M2) tensors, and GGUF's Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K weights; broadcasting; reverse-mode autograd; autocast and loss scaling |
| Layers | linear, convolutions (1-D, 2-D, 3-D, grouped, depthwise, transposed), pooling, normalisation, attention (fused), RoPE, embeddings, LSTM and GRU, graph convolution, over 20 activations |
| Losses | cross-entropy, NLL, MSE, BCE, KL, L1, Huber, hinge, Poisson, CTC, triplet, contrastive, cosine embedding, margin ranking, and more |
| Optimisers | SGD (momentum, Nesterov), Adam, AdamW, AMSGrad, AdaGrad, RMSProp, AdaDelta, AdaMax, NAdam, LAMB, LARS; schedules, clipping, loss scaling |
| Models | ResNet-18/50, MobileNetV2, GPT-2, a Llama-style decoder, BERT, VAE, DCGAN, U-Net, character RNNs; word2vec, doc2vec, GloVe |
| Linear algebra and FFT | `Linalg` (LU, solves, inverse, determinants, Cholesky, QR, symmetric eigendecomposition, SVD) and `Fft` (fft, rfft, 2-D, stft, istft), all differentiable |
| Sparse | sparse x dense products with learnable values, GCN adjacency |
| Data-dependent sizes | nonzero, masked select, unique |
| Classical machine learning | k-means, kNN, PCA, linear and logistic regression, linear SVM, naive Bayes, decision trees, random forests, gradient boosting, t-SNE |
| Several devices | collectives (all-reduce, broadcast, reduce-scatter, all-gather) and data-parallel training across any mix of devices and backends; on NVIDIA GPUs through NCCL inside each rank's plan (1.9x on two GPUs), with synchronised BatchNorm and ranks in several processes |
| Runtime specialisation | plans cached per shape, shape buckets for serving, plan reuse after a reload, and run-time autotuning of kernel variants, saved per device (`-Djitnn.autotune`) |
| ONNX | `jitnn-onnx` loads ONNX files (about 110 operators) and runs them on jitNN's kernels, no third-party dependencies |
| GGUF | `jitnn-gguf` reads GGUF files (llama.cpp's format): metadata, and weights as tensors memory-mapped without a copy, in their quantised blocks or as F32, F16 and BF16 |
| Language models | `jitnn-llm` (optional) generates text from GGUF models through jitllm, with its tokenizers, KV cache and kernels |
| TensorFlow | `jitnn-tf` loads SavedModels and frozen GraphDefs through the ONNX importer's operators |
| DJL | `jitnn-djl` is a Deep Java Library engine: DJL's NDArrays, blocks, `Trainer` and `Predictor` run on jitNN |
| Custom operators | `CustomOp`: a `KernelContext` kernel in plain Java, with an optional host reference and gradient, compiled into the same plans as the built-in operators |
| Vendor libraries | optional: with `--with-native`, products, convolutions, softmax, attention, FFTs and sparse products call cuBLAS, cuDNN, cuFFT, cuSPARSE or MLX inside the same plans |

The full list, with kernels and gradients, is in [Operators](OPERATORS.md).

## Modules

| Module | Contents |
|---|---|
| `jitnn-core` | tensors, autograd, the execution engine, every operator and kernel, `Linalg`, `Fft`, `SparseMatrix`, FP8, collectives |
| `jitnn-nn` | modules and layers, optimisers, the models, embedding models and text sampling, the classical estimators (`ai.beestream.jitnn.ml`), data-parallel training |
| `jitnn-onnx` | the ONNX importer: `OnnxSession` |
| `jitnn-gguf` | the GGUF reader: `GgufFile` |
| `jitnn-llm` | optional (`-Pllm`): text generation from GGUF language models through jitllm, `Llm` |
| `jitnn-djl` | a Deep Java Library engine, named `jitNN` (rank 0, so the default when on the class path): DJL code (NDArray, blocks, Trainer, Predictor) runs on jitNN |
| `jitnn-tf` | the TensorFlow importer: SavedModels and frozen GraphDefs, `TfModel` |
| `jitnn-bench` | kernel and model benchmarks: every operator call of the reference workloads, timed on a device |

## Where it runs

| Platform | Backend | Status |
|---|---|---|
| NVIDIA GPU | CUDA | validated and benchmarked (A10, A100, RTX 5070 Ti) |
| NVIDIA GPU | OpenCL | validated (A100) |
| Apple Silicon | Metal | validated and benchmarked (M4 Pro) for the core, CNN and transformer operators; GEMM on simdgroup matrix units |
| Apple Silicon | OpenCL | validated (M4 Pro) for the core, CNN and transformer operators |
| Other OpenCL GPUs | OpenCL | expected to work, not tested |
| Any CPU | plain JVM | double-precision reference backend, the test oracle |

## Quick start

You need JDK 22+ and a TornadoVM SDK built from `develop` (7.3.2-dev or newer), installed in your
local Maven repository. The full guide is in [Install](INSTALL.md).

```bash
source <TornadoVM>/setvars.sh                        # JAVA_HOME and TORNADOVM_HOME
git clone https://github.com/beestream-ai/jitNN && cd jitnn
./mvnw install -DskipTests
./mvnw test                                          # reference backend, no GPU needed
./mvnw test -Paccel-tests -Djitnn.test.device=0:0    # a TornadoVM device (tornado --devices)
```

Training a model from Java (an MLP here; `nn.models` has ResNet, MobileNetV2, GPT-2, a Llama-style
decoder, BERT and more):

```java
Sequential model = new Sequential(
        new Linear(784, 256, 1), new ReLU(),
        new Linear(256, 128, 2), new ReLU(),
        new Linear(128, 10, 3));
Adam opt = new Adam(model.parameters(), 1e-3f);

for (int step = 0; step < steps; step++) {
    Tensor loss = F.crossEntropy(model.forward(Tensor.of(images, 128, 784)), Tensor.of(labels, 128));
    loss.backward();
    opt.step();
    float value = loss.item();      // sync point: compiles the step once, then replays it
    opt.zeroGrad();
}
```

Running an ONNX model:

```java
OnnxSession session = OnnxSession.load(Path.of("resnet50.onnx"));
float[] logits = session.run(Map.of("input", Tensor.of(images, 1, 3, 224, 224))).get("output").toFloatArray();
```

Run it on the TornadoVM JVM, choosing the device:

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp <jitnn jars>:<your app> -Djitnn.device=1:0 your.Main
```

Add `-Djitnn.native=auto` to let operators call cuBLAS, cuDNN, cuFFT and cuSPARSE (or MLX on Metal)
where they are installed; everything else keeps jitNN's kernels ([Vendor libraries](NATIVE.md)).

Tensors, autograd, sync points, modules, optimisers, and examples for every area above are in
[Usage](USAGE.md).

## Performance

Inference against PyTorch 2.14 and ONNX Runtime on the same GPU, model and inputs: host input in,
host output out, median ms of 50 calls (2026-10-23). jitNN with vendor libraries
(`-Djitnn.native=auto`), fusion on; FP16 rows are FP16 end to end in every framework.

| Workload | A100: jitNN | PyTorch | ONNX Runtime | RTX 5070 Ti: jitNN | PyTorch | ONNX Runtime |
|---|---|---|---|---|---|---|
| ResNet-50 b1 FP32 | **3.40** | 8.12 | 4.05 | **1.98** | 3.91 | 3.18 |
| ResNet-50 b32 FP32 | **11.3** | 31.7 | 29.3 | **14.8** | 22.3 | 18.7 |
| ResNet-50 b128 FP32 | **37.8** | 87.1 | 78.7 | **58.8** | 93.9 | 78.7 |
| ResNet-50 b1 FP16 | **1.60** | 9.78 | 3.57 | **0.88** | 1.72 | 0.90 |
| ResNet-50 b32 FP16 | **7.81** | 14.6 | 8.16 | 9.06 | 11.4 | **7.6** |
| ResNet-50 b128 FP16 | 26.3 | 28.3 | **20.5** | 35.3 | 50.5 | **34.2** |
| BERT-base b1 x 128 FP32 | 3.90 | 5.56 | **3.59** | 3.00 | 2.95 | **2.40** |
| BERT-base b32 x 128 FP32 | 50.2 | **45.7** | 53.6 | 41.4 | **36.8** | 39.5 |
| BERT-base b8 x 512 FP32 | 55.4 | **49.2** | 60.4 | 47.4 | **40.8** | 45.5 |
| BERT-base b1 x 128 FP16 | **1.49** | 6.13 | 2.59 | **0.95** | 1.35 | 1.28 |
| BERT-base b32 x 128 FP16 | 7.27 | **6.24** | 10.4 | 11.9 | **9.8** | 14.1 |
| BERT-base b8 x 512 FP16 | 8.19 | **6.70** | 12.0 | 12.6 | **10.5** | 17.0 |

jitNN is the fastest on 6 of 12 workloads on the A100 and 5 on the RTX 5070 Ti, and at most 1.28x
behind the faster of PyTorch and ONNX Runtime on the rest. On an Apple M4 Pro with MLX, ResNet-50
runs 1.2-1.3x faster than PyTorch MPS and BERT-base at its level; ONNX Runtime with CoreML is
1.1-1.4x ahead on ResNet-50.

On its own kernels (no vendor libraries) jitNN is level with PyTorch on ResNet-50 FP32 and
1.3-3x behind on BERT and FP16. Fusion removes 38% of ResNet-50's kernels and 45% of its memory
traffic; autotuning makes BERT-base 15-34% faster. The baseline element-wise kernels reach 86-88%
of peak bandwidth on CUDA and Metal, reductions and softmax 83-84% on CUDA, and SGEMM 12.7 TFLOP/s
on an RTX 5070 Ti. Training, per-kernel, fusion, autotuning and multi-GPU results are in
[Performance](PERFORMANCE.md).

```{toctree}
:caption: Getting started
:hidden:

INSTALL
USAGE
EXAMPLES
CONFIGURATION
```

```{toctree}
:caption: Reference
:hidden:

OPERATORS
NATIVE
CPU_BACKEND
api
```

```{toctree}
:caption: Correctness and performance
:hidden:

VALIDATION
PERFORMANCE
```

```{toctree}
:caption: Contributing
:hidden:

DEVELOPMENT
```

## License

jitNN is released under the [Apache License, Version 2.0](https://github.com/beestream-ai/jitNN/blob/main/LICENSE).
Copyright (c) 2026 Beestream Ltd.
