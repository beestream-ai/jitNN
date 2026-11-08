<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/jitnn-logo-dark.svg">
    <img alt="jitNN" src="docs/images/jitnn-logo-light.svg" width="480">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Java 22+" src="https://img.shields.io/badge/java-22%2B-orange.svg">
  <img alt="TornadoVM" src="https://img.shields.io/badge/TornadoVM-8.1.0-0B1F33.svg">
  <a href="https://beestream-ai.github.io/jitNN/"><img alt="Javadoc" src="https://img.shields.io/badge/javadoc-public%20API-4c1.svg"></a>
</p>

# jitNN

**jitNN** is a machine-learning and deep-learning library for the JVM, written in Java on
[TornadoVM](https://github.com/beehive-lab/TornadoVM). Tensors, autograd, layers and optimisers
are plain Java; every operator is a Java kernel that TornadoVM JIT-compiles for the GPU it runs on.

- **Portable.** Every operator has a `KernelContext` kernel, forward and backward, so the same code
  runs on CUDA, OpenCL and Metal. Backend-specific variants (tensor-core GEMMs on NVIDIA, simdgroup
  matrices on Apple) sit behind the same operator.
- **Compiled whole.** Operators are recorded lazily. At a sync point the forward pass, the backward
  pass and the optimiser step compile into one TornadoVM task graph, which is cached by structure
  and replayed on every later step. Parameters stay on the device.
- **Validated.** Every operator is checked against PyTorch, forward and backward, on every device,
  and every gradient against finite differences. Models trained from PyTorch's initial weights end
  on PyTorch's losses; the classical estimators match scikit-learn and imported models match ONNX
  Runtime.
- **Open to existing models and code.** ONNX, TensorFlow and GGUF files load directly, and DJL
  programs run unchanged with jitNN as their engine.

## Architecture

<p align="center">
  <img alt="jitNN's layers: your code; jitnn-nn, the importers and the DJL engine; the core API and runtime; the engine that records, rewrites, compiles and replays each step; operators, kernels and vendor libraries; TornadoVM; the hardware" src="docs/images/architecture.svg" width="900">
</p>

Programs, imported models and DJL code all end up as calls on `Tensor` and `F`. The engine records
those calls instead of running them. At a sync point it rewrites the recorded step (fusing
epilogues, folding BatchNorm, reusing buffers), compiles it into one TornadoVM task graph and keeps
it, keyed by the step's structure; later steps with the same structure replay it. Each operator
brings its own kernels, or calls a vendor library where one is enabled. Without TornadoVM, the CPU
backend runs the same operators with the Java Vector API. [DEVELOPMENT.md](docs/DEVELOPMENT.md)
walks through the code.

## What is in it

| Area | Contents |
|---|---|
| Tensors and autograd | FP32, INT32, FP16, BF16 and FP8 tensors, GGUF quantised weights (Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K); broadcasting; reverse-mode autograd; autocast and loss scaling |
| Layers and losses | linear, convolutions (1-D to 3-D, grouped, depthwise, transposed), pooling, normalisation, fused attention, RoPE, embeddings, LSTM and GRU, graph convolution, over 20 activations and losses |
| Optimisers | SGD, Adam, AdamW, AMSGrad, AdaGrad, RMSProp, AdaDelta, AdaMax, NAdam, LAMB, LARS; schedules and clipping |
| Models | ResNet-18/50, MobileNetV2, GPT-2, a Llama-style decoder, BERT, VAE, DCGAN, U-Net, character RNNs, word2vec, doc2vec, GloVe |
| Maths | `Linalg` (LU, solves, inverse, Cholesky, QR, eigendecomposition, SVD), `Fft` (1-D, 2-D, real, STFT), sparse products; all differentiable |
| Classical ML | k-means, kNN, PCA, linear and logistic regression, linear SVM, naive Bayes, decision trees, random forests, gradient boosting, t-SNE |
| Several devices | collectives and data-parallel training across devices and processes, through NCCL on NVIDIA, with synchronised BatchNorm |
| Importers | ONNX (`jitnn-onnx`), TensorFlow SavedModels and GraphDefs (`jitnn-tf`), GGUF (`jitnn-gguf`), text generation from GGUF models through jitllm (`jitnn-llm`, optional) |
| DJL | `jitnn-djl`: a Deep Java Library engine; DJL's NDArrays, blocks, `Trainer` and `Predictor` run on jitNN |
| Extensions | `CustomOp`: your own `KernelContext` kernel, with an optional gradient, compiled into the same plans |

Every operator, with its kernels and gradient, is listed in [OPERATORS.md](docs/OPERATORS.md).

## Modules

| Module | Contents |
|---|---|
| `jitnn-core` | tensors, autograd, the engine, every operator and kernel, `Linalg`, `Fft`, `SparseMatrix`, collectives |
| `jitnn-nn` | layers, optimisers, models, data loaders, tokenizers, classical estimators (`ai.beestream.jitnn.ml`) |
| `jitnn-onnx` | the ONNX importer and exporter: `OnnxSession` |
| `jitnn-tf` | the TensorFlow importer: `TfModel` |
| `jitnn-gguf` | the GGUF reader: `GgufFile` |
| `jitnn-llm` | optional (`-Pllm`): text generation through jitllm, `Llm` |
| `jitnn-djl` | the DJL engine, named `jitNN` (the default engine when it is on the class path) |
| `jitnn-examples` | runnable examples after DL4J, Deep Netts, DJL, PyTorch, Keras, H2O, Tribuo and Smile ([README](jitnn-examples/README.md)) |

## Where it runs

| Hardware | Backend | Status |
|---|---|---|
| NVIDIA GPUs | CUDA (TornadoVM) | validated on Ampere and Blackwell (A10, A100, RTX 5070 Ti); vendor libraries: cuBLAS, cuDNN, cuFFT, cuSPARSE, CUTLASS, NCCL |
| NVIDIA GPUs | OpenCL (TornadoVM) | validated (A100) |
| Apple silicon | Metal (TornadoVM) | validated (M4 Pro) for the core, CNN and transformer operators; vendor library: MLX |
| Apple silicon | OpenCL (TornadoVM) | validated (M4 Pro) for the core, CNN and transformer operators |
| Other OpenCL GPUs | OpenCL (TornadoVM) | expected to work, not tested |
| Any CPU | `cpu` backend | Java Vector API kernels, multithreaded; optional BLAS, oneDNN or BNNS ([CPU_BACKEND.md](docs/CPU_BACKEND.md)) |
| Any CPU | `reference` backend | scalar Java with double accumulation: the oracle the tests check against |

The backend is `tornado` when a TornadoVM runtime is present, else `cpu` when the JVM runs with
`--add-modules jdk.incubator.vector`, else `reference`; `-Djitnn.backend` overrides it. Pick the
TornadoVM device with `-Djitnn.device=<driver>:<device>` (`tornado --devices` lists them).

## Quick start

You need JDK 22 or newer and a TornadoVM 8.1.0 SDK to run on; Maven fetches TornadoVM's API from
Maven Central. [INSTALL.md](docs/INSTALL.md) has the details.

```bash
source <TornadoVM>/setvars.sh                        # JAVA_HOME and TORNADOVM_HOME
git clone https://github.com/beestream-ai/jitNN && cd jitNN
./mvnw install -DskipTests
./mvnw test                                          # reference backend, no GPU needed
./mvnw test -Paccel-tests -Djitnn.test.device=0:0    # on a TornadoVM device
```

Training a model:

```java
Sequential model = new Sequential(
        new Linear(784, 256, 1), new ReLU(),
        new Linear(256, 10, 2));
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

Run on the TornadoVM JVM:

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp <jitnn jars>:<your app> -Djitnn.device=1:0 your.Main
```

[USAGE.md](docs/USAGE.md) covers every area above with examples; the
[examples](jitnn-examples/README.md) are complete programs to start from.

## Performance flags

The defaults are safe and portable. These options trade portability, precision or memory for speed;
all are JVM system properties (`-D...`) unless marked as an API call.

| Option | What it does | Use it when |
|---|---|---|
| `-Djitnn.native=auto` | calls cuBLAS, cuDNN, cuFFT, cuSPARSE and CUTLASS on NVIDIA, MLX on Apple, for the operators they cover; anything missing falls back to jitNN's kernels ([NATIVE.md](docs/NATIVE.md)) | the libraries are installed: usually the largest single gain for convolutions, products and attention |
| `-Djitnn.native.tf32=true` | FP32 products in cuBLAS run on TF32 tensor cores; with `native=auto`, FP32 convolutions go to cuDNN only when this is on | training or inference that tolerates TF32's 10-bit mantissa, as PyTorch's convolutions do by default |
| `Autocast.enable(DType.F16)` or `DType.BF16` (API) | products in half precision, everything else in FP32 | training on GPUs with tensor cores; pair FP16 with loss scaling (`GradScaler`) |
| `Autocast.enableHalf()` (API) | FP16 end to end for inference | inference on tensor cores or Apple GPUs |
| `-Djitnn.autotune=on` | times the kernel variants of each new GEMM, convolution and row-norm shape once and keeps the fastest; decisions are saved per device in `~/.jitnn/autotune` | a model you run many times on the same GPU; later runs can use `-Djitnn.autotune=use` to apply saved decisions without measuring |
| `-Djitnn.concurrency=N` | N slots, each with its own compiled plans and device buffers, so N threads run on the device at once | serving many requests; costs N copies of the weights in device memory |
| `-Djitnn.onnx.buckets=1,2,4,8,...` or `OnnxSession.withBuckets` | pads symbolic batch and sequence sizes to a few buckets, one compiled plan per bucket | serving inputs of varying size without compiling each shape |
| `-Djitnn.plan.cache=N` | how many compiled plans are kept (default 16) | a workload alternating between more shapes or models than that |
| `-Dtornado.device.memory=<N>GB` | TornadoVM's device-memory budget (TornadoVM's default is 4GB) | large models or batches |
| `-Djitnn.backend=cpu`, `-Djitnn.cpu.threads=N` | the Vector API CPU backend and its thread count; it uses MKL, OpenBLAS, Accelerate, oneDNN or BNNS when found ([CPU_BACKEND.md](docs/CPU_BACKEND.md)) | no GPU; install a BLAS and oneDNN for the convolution- and product-heavy models |

Fusion, buffer reuse, channels-last inference, Winograd convolutions, CUDA graphs and kernel
sharing across shapes are on by default; their switches exist for debugging. In your own code, keep
shapes stable between steps, read results (`item()`, `toFloatArray()`) once per step, since every
read is a sync point, and release what you no longer need: tensors with `release()`, models with
`close()`, which also frees their derived weights and idle plans. To see where time
goes, run with `-Djitnn.profile=true` and print `JitnnRuntime.kernelBreakdown()`; `-Djitnn.verbose=true`
logs every compiled plan and `-Djitnn.debugKeys=true` shows why a step recompiles.
[CONFIGURATION.md](docs/CONFIGURATION.md) lists every option. Benchmarks and results are in
[jitNN-benchmarking](https://github.com/beestream-ai/jitNN-benchmarking).

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | TornadoVM, building, test data, the Python environments for the references |
| [USAGE.md](docs/USAGE.md) | the user guide: tensors, sync points, threads and errors, autograd, modules, training, maths, devices, importers, DJL, custom operators |
| [EXAMPLES](jitnn-examples/README.md) | complete example programs and how to run them |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | every `jitnn.*` option |
| [NATIVE.md](docs/NATIVE.md) | vendor libraries: which operators use them, and the fallback |
| [CPU_BACKEND.md](docs/CPU_BACKEND.md) | the Vector API CPU backend |
| [OPERATORS.md](docs/OPERATORS.md) | every operator, the ONNX and TensorFlow operators, the DJL engine |
| [VALIDATION.md](docs/VALIDATION.md) | how correctness is checked: references, gradients, model gates |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | architecture, the public API and the internal packages, code layout, adding an operator, code generators |
| [API reference](https://beestream-ai.github.io/jitNN/) | Javadoc of the public API of every module ([api.md](docs/api.md): the packages and building it) |

The same pages build as a Read the Docs site: `pip install -r docs/requirements.txt && make -C docs html`.

## License

jitNN is released under the [Apache License, Version 2.0](LICENSE). Copyright (c) 2026 Beestream Ltd.
Third-party material (code from jitllm under the MIT License, the Iris data under CC BY 4.0) is
listed in [NOTICE](NOTICE).
