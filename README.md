# jitNN

jitNN is a machine-learning and deep-learning library for the JVM. Every operator is a
[TornadoVM](https://github.com/beehive-lab/TornadoVM) kernel written in Java, so the same code runs on
OpenCL, CUDA and Metal devices. Its operator set is the union of what DL4J/ND4J, DJL and Deep Netts
provide, built wave by wave in order of how often real models use each operator.

- **KernelContext baseline.** Every operator has a portable KernelContext kernel (forward and
  backward). CUDA-only optimisations (TileContext kernels, cuBLAS/cuDNN library tasks) come later,
  behind the same operator interface, and are validated against the baseline.
- **Traced and replayed.** Operators are recorded lazily. A sync point compiles everything since the
  last one (forward, backward and optimiser step) into one TornadoVM task graph, and caches it by
  structure: the second and later training steps replay the compiled plan. Parameters stay on the
  device between steps.
- **Validated.** Every operator has a double-precision Java reference. Tests check the reference
  and every device against PyTorch golden tensors (forward and gradients), check every gradient
  against finite differences, and train an MNIST MLP from the same initial weights as PyTorch.

Status: wave 1 (core operators and the MLP) done at the baseline; its performance pass is next.
Validated on OpenCL and Metal (Apple M4 Pro) and CUDA (RTX 5070 Ti).

## Build and test

You need JDK 22+ and a TornadoVM SDK built from `develop` (7.0.2-dev, which has the on-demand
`TornadoExecutionPlan.transferToDevice(Object...)`), installed in the local Maven repository.

```bash
source <TornadoVM>/setvars.sh          # sets JAVA_HOME and TORNADOVM_HOME
./mvnw test                            # host reference tests, no accelerator needed
./mvnw test -Paccel-tests              # device tests on TornadoVM device 0:0
./mvnw test -Paccel-tests -Djitnn.test.device=1:0   # another device (tornado --devices)
```

### PyTorch references

`scripts/pytorch` needs Python with `torch` and `numpy`.

```bash
python scripts/pytorch/goldens.py jitnn-core/src/test/resources/golden   # regenerate the golden set
python scripts/pytorch/mnist_mlp.py <mnist dir> <mnist dir>/mnist_mlp_pytorch.f32
./mvnw test -Paccel-tests -Djitnn.mnist=<mnist dir>   # MNIST gate against that PyTorch run
```

`jitnn_init.py` reproduces jitNN's Philox initialisation in numpy, so PyTorch starts from the same
weights.

## Example

```java
Tensor x = Tensor.of(data, batch, 784);
Tensor labels = Tensor.of(classes, batch);
Tensor w = Tensor.parameter(Tensor.normal(1, 0, 0.05f, 10, 784));
Tensor b = Tensor.parameter(Tensor.zeros(10));

Tensor loss = F.crossEntropy(F.linear(x, w, b), labels);
loss.backward();
float value = loss.item();   // sync point: compiles once, replays afterwards
```

## Layout

| Module | Contents |
|---|---|
| `jitnn-core` | Tensor, autograd, trace compiler and executor, operators, KernelContext kernels, host reference |
| `jitnn-nn` | Modules, layers, losses and optimisers |

## Options

| Property | Default | Meaning |
|---|---|---|
| `jitnn.backend` | `tornado` if a TornadoVM runtime is present, else `reference` | Where operators run |
| `jitnn.device` | TornadoVM default | `<driver>:<device>` for compiled plans |
| `jitnn.plan.cache` | 16 | Compiled plans kept before the least recently used is closed |
| `jitnn.verbose` | false | Log each compiled plan |

## Known TornadoVM workarounds

Kernels tagged `WORKAROUND(...)` avoid TornadoVM issues; each has a test that pins the behaviour.

- Philox RNG avoids `>>>` (TornadoVM#1140).
- Nested bounds checks that Graal can prove constant crash OpenCL code generation
  (`LogicConstantNode` in `emitIf`); the RNG kernel uses one thread per element.
- Kernel conditions use non-short-circuit `&`/`|` (short-circuit operators in loops can drop a
  branch).
- Accelerator tests run without `-ea`: TornadoVM's Graal plugin checks fail under assertions on
  JDK 22+.

## License

Apache License 2.0.
