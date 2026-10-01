<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/jitnn-logo-dark.svg">
    <img alt="jitNN" src="docs/images/jitnn-logo-light.svg" width="480">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Java 22+" src="https://img.shields.io/badge/java-22%2B-orange.svg">
  <img alt="TornadoVM" src="https://img.shields.io/badge/TornadoVM-7.0.2--dev-0B1F33.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-wave%202-22D3A6.svg">
</p>

# jitNN

**jitNN** is a machine-learning and deep-learning library for the JVM, written in Java on
[TornadoVM](https://github.com/beehive-lab/TornadoVM). Tensors, autograd, layers and optimisers
are plain Java, and every operator is a TornadoVM kernel that is JIT-compiled for the GPU it runs
on. Its operator set is the union of what DL4J/ND4J, DJL and Deep Netts provide, built wave by wave
in the order real models use it.

- **Portable.** Every operator has a KernelContext kernel, forward and backward, so the same code
  runs on CUDA, OpenCL and Metal. Faster backend-specific variants (Metal simdgroup matrices, and on
  NVIDIA, CUDA Tile and cuBLAS/cuDNN later on) sit behind the same operator and are checked against
  the baseline.
- **Compiled whole.** Operators are recorded lazily. At a sync point the forward pass, the backward
  pass and the optimiser step compile into one TornadoVM task graph, which is cached by structure
  and replayed on every following training step, as a CUDA graph on NVIDIA. Parameters stay on the
  device between steps.
- **Validated against PyTorch.** Every operator is checked against PyTorch golden tensors, forward
  and gradients, on every device, and every gradient against finite differences. Models are
  trained from the same initial weights as PyTorch: an MNIST MLP tracks its loss to 1e-6, and
  ResNet-18 and MobileNetV2 match PyTorch's FP64 gradients as closely as PyTorch's own FP32 does,
  and after one CIFAR-10 epoch reach PyTorch's test accuracy (66-67% and 58-61%).
- **Fused and frugal.** Bias and ReLU run in the GEMM and convolution epilogues, cross-entropy's
  backward is a single kernel, and dead backward-pass buffers are reused inside a compiled step
  (-43% memory on ResNet-18). On CUDA the element-wise, reduction and softmax kernels reach 83-88%
  of peak memory bandwidth.

## Where it runs

| Platform | Backend | Status |
|---|---|---|
| NVIDIA GPU | CUDA | validated and benchmarked (RTX 5070 Ti) |
| Apple Silicon | Metal | validated and benchmarked (M4 Pro); GEMM on simdgroup matrix units |
| Apple Silicon | OpenCL | validated (M4 Pro) |
| Other OpenCL GPUs | OpenCL | expected to work, not tested |
| Any CPU | plain JVM | double-precision reference backend, the test oracle |

## Quick start

You need JDK 22+ and a TornadoVM SDK built from `develop` (7.0.2-dev), installed in your local Maven
repository. The full guide is in [docs/INSTALL.md](docs/INSTALL.md).

```bash
source <TornadoVM>/setvars.sh                        # JAVA_HOME and TORNADOVM_HOME
git clone https://github.com/beehive-lab/jitnn && cd jitnn
./mvnw install -DskipTests
./mvnw test                                          # reference backend, no GPU needed
./mvnw test -Paccel-tests -Djitnn.test.device=0:0    # a TornadoVM device (tornado --devices)
```

Training a model from Java (an MLP here; `nn.models` has ResNet-18 and MobileNetV2):

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

Run it on the TornadoVM JVM, choosing the device:

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp <jitnn jars>:<your app> -Djitnn.device=1:0 your.Main
```

Tensors, autograd, sync points, modules and optimisers are covered in [docs/USAGE.md](docs/USAGE.md).

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

On Apple GPUs a step is dominated by TornadoVM's per-kernel dispatch cost (about 0.15 ms per
kernel, 268 kernels in a ResNet-18 step), and the convolutions share the scalar GEMM's ~2 TFLOP/s
ceiling; both are the next performance targets. The CUDA ResNet-18 figure is without a pending
TornadoVM fix for reading results from multi-graph plans. Method, all kernels, and the remaining gaps are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | TornadoVM SDK, building, test data, the PyTorch environment for references |
| [USAGE.md](docs/USAGE.md) | Tensors, lazy execution and sync points, autograd, modules, optimisers, the training loop |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every `jitnn.*` option, kernel variants, test properties |
| [OPERATORS.md](docs/OPERATORS.md) | The wave 1 and 2 operators, their kernels, and the roadmap of waves |
| [VALIDATION.md](docs/VALIDATION.md) | Reference backend, finite differences, PyTorch goldens, the MNIST and CNN gates |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | Per-kernel and end-to-end results on CUDA, Metal and OpenCL, against vendor libraries |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, code layout, adding an operator, code generators, TornadoVM workarounds |

## License

jitNN is released under the [Apache License, Version 2.0](LICENSE).
Copyright (c) 2026 Beestream Ltd.
