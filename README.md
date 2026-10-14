<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/jitnn-logo-dark.svg">
    <img alt="jitNN" src="docs/images/jitnn-logo-light.svg" width="480">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Java 22+" src="https://img.shields.io/badge/java-22%2B-orange.svg">
  <img alt="TornadoVM" src="https://img.shields.io/badge/TornadoVM-develop-0B1F33.svg">
</p>

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
| Several devices | collectives (all-reduce, broadcast, reduce-scatter, all-gather) and data-parallel training across any mix of devices and backends |
| ONNX | `jitnn-onnx` loads ONNX files (about 110 operators) and runs them on jitNN's kernels, no third-party dependencies |
| TensorFlow | `jitnn-tf` loads SavedModels and frozen GraphDefs through the ONNX importer's operators |
| DJL | `jitnn-djl` is a Deep Java Library engine: DJL's NDArrays, blocks, `Trainer` and `Predictor` run on jitNN |
| Custom operators | `CustomOp`: a `KernelContext` kernel in plain Java, with an optional host reference and gradient, compiled into the same plans as the built-in operators |
| Vendor libraries | optional: with `--with-native`, products, convolutions, softmax, attention, FFTs and sparse products call cuBLAS, cuDNN, cuFFT, cuSPARSE or MLX inside the same plans |

The full list, with kernels and gradients, is in [docs/OPERATORS.md](docs/OPERATORS.md).

## Modules

| Module | Contents |
|---|---|
| `jitnn-core` | tensors, autograd, the execution engine, every operator and kernel, `Linalg`, `Fft`, `SparseMatrix`, FP8, collectives |
| `jitnn-nn` | modules and layers, optimisers, the models, embedding models and text sampling, the classical estimators (`org.beehive.jitnn.ml`), data-parallel training |
| `jitnn-onnx` | the ONNX importer: `OnnxSession` |
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

You need JDK 22+ and a TornadoVM SDK built from `develop` (7.2.1-dev or newer), installed in your
local Maven repository. The full guide is in [docs/INSTALL.md](docs/INSTALL.md).

```bash
source <TornadoVM>/setvars.sh                        # JAVA_HOME and TORNADOVM_HOME
git clone https://github.com/beehive-lab/jitnn && cd jitnn
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
where they are installed; everything else keeps jitNN's kernels ([docs/NATIVE.md](docs/NATIVE.md)).

Tensors, autograd, sync points, modules, optimisers, and examples for every area above are in
[docs/USAGE.md](docs/USAGE.md).

## Performance

KernelContext baseline. Kernel times come from the TornadoVM profiler; training rows are end to
end (MNIST, Adam).

| | RTX 5070 Ti (CUDA) | M4 Pro (Metal) | M4 Pro, PyTorch MPS |
|---|---|---|---|
| Element-wise, 16M elements | 790 GB/s (88% of peak) | 235 GB/s (86%) | add: 239 GB/s |
| Reductions and softmax | 740-755 GB/s | 120-200 GB/s | softmax: 183 GB/s |
| SGEMM 2048 | 12.7 TFLOP/s | 2.3 TFLOP/s | 5.4 TFLOP/s |
| Conv 3x3, 64 channels, 28x28, batch 64 | | 1.9 TFLOP/s | 6.1 TFLOP/s |
| MLP training, batch 128 | 210k samples/s | 28.7k samples/s | |
| ResNet-18 training, batch 64 | 160 ms/step | 165 ms/step | 54 ms/step |
| MobileNetV2 training, batch 64 | | 148 ms/step | 62 ms/step |
| GPT-2 style training, 16 x 128 tokens, 0.84M parameters | 12.7 ms/step (A10; 6.7 with BF16) | 87 ms/step | |

Inference on an A10 against PyTorch 2.11 and ONNX Runtime 1.30 on the same model files (CUDA,
median ms), jitNN with its own kernels and with vendor libraries (`--with-native`):

| Model | PyTorch | ONNX Runtime | jitNN kernels | jitNN `--with-native` |
|---|---|---|---|---|
| ResNet-50 b1 FP32 | 4.93 | 2.98 | 4.05 | 4.04 |
| ResNet-50 b32 FP32 | 43.9 | 39.1 | 52.2 | 33.3 |
| ResNet-50 b128 FP32 | 168.2 | 145.7 | 217.9 | 126.9 |
| BERT-base b1 x 128 FP32 | 3.67 | 4.08 | 7.08 | 4.11 |
| BERT-base b32 x 128 FP32 | 79.2 | 86.1 | 125.6 | 82.4 |
| BERT-base b8 x 512 FP32 | 89.2 | 98.2 | 139.2 | 96.0 |
| BERT-base b1 x 128 FP16 | 3.44 | 1.82 | 4.79 | 2.95 |
| BERT-base b32 x 128 FP16 | 16.3 | 27.0 | 48.3 | 38.5 |
| BERT-base b8 x 512 FP16 | 17.3 | 31.3 | 61.1 | 51.4 |

In FP32 with vendor libraries, ResNet-50 at batch 32 and 128 is ahead of both, and BERT-base is
at or ahead of ONNX Runtime and within 8% of PyTorch. FP16 is behind:
jitNN's FP16 is autocast (FP16 weights and products, FP32 activations, attention and norms), while
PyTorch and ONNX Runtime run the whole model in half precision. On its own kernels jitNN is about
1.5x behind ONNX Runtime at large batch in FP32, set by the speed of its convolution and GEMM
kernels against cuDNN and cuBLAS.

On Apple GPUs a training step is dominated by TornadoVM's per-kernel dispatch cost (about 0.15 ms
per kernel, 268 kernels in a ResNet-18 step), and the convolutions share the scalar GEMM's
~2 TFLOP/s ceiling. The RTX 5070 Ti ResNet-18 figure predates a TornadoVM fix for reading results
from multi-graph plans; on an A10 with it, a ResNet-18 step takes 66 ms and a MobileNetV2 step
47 ms. Method, all results (including the earlier A100 inference comparison) and the remaining
gaps are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | TornadoVM SDK, building, test data, the Python environments for references |
| [USAGE.md](docs/USAGE.md) | Tensors, lazy execution and sync points, autograd, modules, optimisers, the training loop, linear algebra, FFT, sparse, embeddings, classical ML, FP8, several devices, ONNX, TensorFlow, DJL, custom operators, printing the kernels, vendor libraries |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every `jitnn.*` option: runtime, kernel variants, importers and DJL, vendor libraries, benchmark and test properties |
| [NATIVE.md](docs/NATIVE.md) | `--with-native`: which operators call cuBLAS, cuDNN, cuFFT, cuSPARSE or MLX, and the fallback to jitNN kernels |
| [OPERATORS.md](docs/OPERATORS.md) | Every operator and its kernels and gradient, the ONNX and TensorFlow operators, the DJL engine, what is not supported yet |
| [VALIDATION.md](docs/VALIDATION.md) | Reference backend, finite differences, PyTorch, scikit-learn and ONNX Runtime references, the model gates |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | Per-kernel, training and inference results on CUDA, Metal and OpenCL, against PyTorch and ONNX Runtime, with and without vendor libraries |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, code layout, adding an operator, code generators, TornadoVM workarounds |

## License

jitNN is released under the [Apache License, Version 2.0](LICENSE).
Copyright (c) 2026 Beestream Ltd.
