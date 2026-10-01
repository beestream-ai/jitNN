# Using jitNN

## Tensors

A `Tensor` is an n-dimensional, contiguous, row-major array of FP32 values or INT32 labels.

```java
Tensor x = Tensor.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);   // data, then shape
Tensor y = Tensor.of(new int[] { 2, 0 }, 2);                     // INT32 labels
Tensor z = Tensor.zeros(4, 4);
Tensor u = Tensor.uniform(42, -1, 1, 128, 784);                  // seed, range, shape
Tensor n = Tensor.normal(42, 0, 0.02f, 512);
```

Random tensors come from a counter-based generator (Philox4x32-10): the same seed gives the same
values on every device and on the host, so a model initialises identically everywhere.

Operators are methods (`add`, `mul`, `exp`, `relu`, `sum(axis, keepDim)`, `max`, `argmax`,
`matmul`, `permute`, `reshape`, `softmax`, `logSoftmax`, ...) and functions in `F` (`linear`,
`linearRelu`, `crossEntropy`, `nll`, `accuracy`). Binary operators broadcast like NumPy.

## Lazy execution and sync points

On a TornadoVM device, operators are recorded, not run. A **sync point** runs everything recorded
on the thread so far:

- `item()`, `toFloatArray()`, `toIntArray()` on any tensor,
- `Engine.sync()`.

At a sync point, jitNN compiles the recorded operators into one TornadoVM task graph and caches the
compiled plan under the trace's structure (operators, shapes, which parameters). When the same
structure comes back, as it does on every training step, the plan is replayed without compiling.

Two consequences follow:

- **Read what you need before the next step.** A replayed plan reuses its buffers. A tensor from
  an earlier step that was never read throws `IllegalStateException` ("data was reused") if you
  read it later. Values already read to the host stay valid.
- **Keep the step's structure stable.** A different batch size or a new branch compiles a new plan.
  `-Djitnn.verbose=true` logs each compile.

## Autograd

```java
Tensor w = Tensor.of(values, 10, 784).requiresGrad(true);
Tensor loss = F.crossEntropy(F.linear(x, w, null), labels);
loss.backward();
Tensor g = w.grad();
```

`backward()` records the gradient operators into the same trace, so on a device the forward pass,
the backward pass and the optimiser step become one compiled graph. `Grad.disabled()` turns
recording off in a try-with-resources block.

## Parameters and modules

A **parameter** has persistent storage: it lives on the device across steps and only moves to the
host when read. `Tensor.parameter(init)` creates one from an initial value. `Module` subclasses
register their parameters:

| Module | Notes |
|---|---|
| `Linear(in, out, seed)` | PyTorch layout and initialisation (uniform in plus or minus 1/sqrt(in)) |
| `ReLU` | |
| `Sequential(...)` | runs `Linear` then `ReLU` as one fused GEMM |

### CNN layers and models

| Module | Notes |
|---|---|
| `Conv2d(in, out, k, stride, padding, seed)` and the full constructor with dilation, groups, bias | NCHW, square kernels; PyTorch initialisation |
| `BatchNorm2d(channels)` | batch statistics in training mode, running statistics in evaluation mode |
| `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `Flatten` | |
| `Dropout(p, seed)` | a new mask every step without recompiling |
| `ReLU6`, `Add(main, shortcut, relu)` | `Add` is a residual connection |
| `ResNet.resnet18(in, classes, small, seed)`, `MobileNetV2.mobilenetV2(in, classes, small, dropout, seed)` | torchvision layouts; `small` adapts the stem to 28-32 pixel inputs |

`Sequential` fuses a ReLU into the Linear, Conv2d or BatchNorm2d before it. Switch modes with
`model.train(false)` for evaluation.

## Optimisers

| Optimiser | Constructor |
|---|---|
| SGD | `new SGD(params, lr)` |
| SGD with momentum, Nesterov, weight decay | `new SGD(params, lr, momentum, weightDecay, nesterov)` |
| Adam | `new Adam(params, lr)` or the full constructor |
| AdamW | `Adam.adamW(params, lr, weightDecay)` |

Updates are in place on the device. The learning rate and Adam's bias corrections are uploaded
with each step, so schedules work without recompiling (`setLearningRate`).

Learning-rate schedules (`LRScheduler.step`, `multiStep`, `cosine`) adjust the optimiser once per
epoch, like PyTorch's.

## The training loop

```java
for (int step = 0; step < steps; step++) {
    Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, batch, 784)), Tensor.of(yb, batch));
    loss.backward();
    opt.step();
    float value = loss.item();     // one compiled plan, replayed from the second step
    opt.zeroGrad();
}
float acc = F.accuracy(model.forward(Tensor.of(testX, n, 784)), Tensor.of(testY, n));
```

## Inference and memory

```java
Tensor logits = Grad.inference(() -> model.forward(x));
```

Inside `Grad.inference` no gradients are recorded, and every intermediate except the returned
tensor gives its buffer back as soon as the computation no longer needs it (an evaluation pass of
MobileNetV2 needs 89% less device memory). Gradient temporaries inside `backward()` are reused the
same way. Forward activations of a training step are kept, so they can still be read after the step.

## Backends

`Engine.backend()` is `TORNADO` when a TornadoVM runtime is on the class path, else `REFERENCE`:
the host executes each operator eagerly in double precision. `Engine.with(Backend.REFERENCE, ...)`
runs a block on the reference, which is how the tests compare the two.

## Tests

| Command | What runs |
|---|---|
| `./mvnw test` | reference backend: values, finite differences, PyTorch goldens, MLP convergence |
| `./mvnw test -Paccel-tests [-Djitnn.test.device=d:i]` | every operator and golden case on a device, the MLP gate, optimiser goldens |
| `... -Djitnn.mnist=<dir>` | adds the MNIST gate against PyTorch |
| `... -Dtest=BaselineBenchAccelTest -Djitnn.test.profile=true` | kernel benchmarks |
