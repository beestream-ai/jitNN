# Using jitNN

## Tensors

A `Tensor` is an n-dimensional, contiguous, row-major array. Its `DType` is `F32` (the default),
`I32` (labels and indices), `F16` or `BF16` (mixed precision), `Q8_0` (GGUF's 8-bit blocks) or
`FP8_E4M3`/`FP8_E5M2` (8-bit floats).

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
| `ResNet.resnet18(in, classes, small, seed)`, `ResNet.resnet50(...)`, `MobileNetV2.mobilenetV2(in, classes, small, dropout, seed)` | torchvision layouts; `small` adapts the stem to 28-32 pixel inputs |

`Sequential` fuses a ReLU into the Linear, Conv2d or BatchNorm2d before it. Switch modes with
`model.train(false)` for evaluation.

### Transformer layers and models

| Module | Notes |
|---|---|
| `Embedding(count, dim, seed)`, with a padding index | lookups by INT32 indices; deterministic gradient |
| `LayerNorm(features)`, `RMSNorm(features, eps)` | over the last axis |
| `MultiheadAttention(dim, heads, bias, seed)` | self-attention on [B, T, D]; `forward(x, mask, causal)` |
| `GPT2`, `Llama`, `Bert` in `nn.models` | a GPT-2 decoder, a Llama-style decoder (RMSNorm, RoPE, SwiGLU) and a BERT encoder with an MLM head |

The functional forms are in `F`: `embedding`, `layerNorm`, `rmsNorm`, `scaledDotProductAttention`
(causal and additive masks), `multiHeadAttention`, `rope`, `swiglu`, `multinomial`, and on
`Tensor`: `gelu`, `geluTanh`, `silu`, `sigmoid`, `tanh`, `narrow`, `slice`, `split`, `chunk`,
`cat`, `stack`, `topk`.

### Generative and recurrent layers and models

| Module | Notes |
|---|---|
| `ConvTranspose2d(in, out, kernel, stride, padding, outputPadding, dilation, groups, bias, seed)` | PyTorch's weight layout [in, out/groups, k, k] and initialisation |
| `Upsample(scale, bilinear, alignCorners)` | nearest or bilinear; `F.interpolate` resizes to a given size |
| `LeakyReLU(slope)`, `Sigmoid`, `Tanh` | |
| `LSTM(input, hidden, layers, seed)`, `GRU(...)` | time-first [T, B, I]; `forward(x, h0, c0)` returns the output sequence and the final states |
| `Vae`, `Dcgan.Generator`, `Dcgan.Discriminator`, `UNet`, `CharRnn` in `nn.models` | |

Losses in `F`: `mseLoss`, `binaryCrossEntropy`, `binaryCrossEntropyWithLogits` (each with a
`Reduction`), `gaussianKl`. Noise drawn every step should take its seed as a tensor, so the compiled
step is replayed rather than recompiled:

```java
Tensor eps = Tensor.uniform(Tensor.seed(step), -1.7320508f, 1.7320508f, batch, latent);
```

## Optimisers

| Optimiser | Constructor |
|---|---|
| SGD | `new SGD(params, lr)` |
| SGD with momentum, Nesterov, weight decay | `new SGD(params, lr, momentum, weightDecay, nesterov)` |
| Adam | `new Adam(params, lr)` or the full constructor |
| AdamW | `Adam.adamW(params, lr, weightDecay)` |
| LAMB | `new Lamb(params, lr, beta1, beta2, eps, weightDecay)` |
| LARS | `new Lars(params, lr, momentum, weightDecay, eta, eps)` |
| AdaGrad, RMSProp, AdaDelta, AdaMax, NAdam | `nn.optim`, with PyTorch's arguments |

Updates are in place on the device. The learning rate and Adam's bias corrections are uploaded
with each step, so schedules work without recompiling (`setLearningRate`).

Learning-rate schedules (`LRScheduler.step`, `multiStep`, `cosine`) adjust the optimiser once per
epoch, like PyTorch's.
`LRScheduler.warmupLinear` and `warmupCosine` count optimiser steps instead (call `step()` after
every `opt.step()`). `opt.clipGradNorm(maxNorm)` clips the gradients by their global norm at the
next step, as PyTorch's `clip_grad_norm_`, inside the update itself (one small reduction per
gradient); `Grad.clipNorm(params, maxNorm)` rescales the gradient tensors themselves, for code
that reads them.

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

## Mixed precision

`Tensor.half()`, `bfloat16()` and `to(DType)` convert between FP32, FP16 and BF16 (round to
nearest even, as PyTorch). Inside an autocast scope the GEMMs (`F.linear`, `matmul` and the two
products of attention) run on half-precision operands with FP32 accumulation; their results and
every other operator stay FP32. FP16 needs dynamic loss scaling, as with PyTorch's `GradScaler`:

```java
GradScaler scaler = new GradScaler();
for (int step = 0; step < steps; step++) {
    Tensor loss;
    try (var ac = Autocast.enable(DType.F16)) {      // or DType.BF16, usually without a scaler
        loss = F.crossEntropy(model.forward(x), y);
    }
    scaler.scale(loss).backward();
    scaler.unscale(opt);                              // before clipping
    opt.clipGradNorm(1.0f);
    scaler.step(opt);                                 // skipped on the device if a gradient overflowed
    float value = loss.item();
    scaler.update();                                  // reads the overflow flag, adjusts the scale
    opt.zeroGrad();
}
```

The scale is uploaded with each step and overflows are detected on the device, so the step stays
one compiled plan. BF16 needs TornadoVM with float-bit reinterpretation (TornadoVM#1141, in
`develop` since 2026-10-02) on OpenCL and Metal.

## Inference and memory

```java
Tensor logits = Grad.inference(() -> model.forward(x));
```

Inside `Grad.inference` no gradients are recorded, and every intermediate except the returned
tensor gives its buffer back as soon as the computation no longer needs it (an evaluation pass of
MobileNetV2 needs 89% less device memory). Gradient temporaries inside `backward()` are reused the
same way. Forward activations of a training step are kept, so they can still be read after the step.

In evaluation mode and inside `Grad.inference`, convolution + BatchNorm (+ ReLU) runs as one
convolution on folded weights, a GELU after a linear layer runs in the GEMM epilogue, and add +
ReLU is one kernel (see [OPERATORS.md](OPERATORS.md#inference-fusion)).

## More losses

Besides cross-entropy, NLL and the losses above, `F` has `ctcLoss` (connectionist temporal
classification), `tripletMarginLoss`, `contrastiveLoss`, `cosineEmbeddingLoss`,
`marginRankingLoss`, the regression and classification losses of PyTorch (L1, Huber, smooth L1,
hinge, Poisson NLL, ...), and `sparsemax` and `logSigmoid`:

```java
// logProbs [T, B, C] after logSoftmax over C; targets [B, maxLabels] INT32, padded
Tensor loss = F.ctcLoss(logProbs, targets, inputLengths, targetLengths, 0, F.Reduction.MEAN);
```

## Linear algebra and Fourier transforms

`Linalg` works on a matrix or a batch of matrices [..., rows, cols], and every function has a
gradient:

```java
Tensor a = Tensor.uniform(1, -1, 1, 64, 8, 8);               // 64 matrices of 8 x 8
Tensor b = Tensor.uniform(2, -1, 1, 64, 8, 3);
Tensor x = Linalg.solve(a, b);                                 // A X = B, [64, 8, 3]
Tensor logDet = Linalg.slogdet(a)[1];                          // [sign, log|det|]
Tensor[] qr = Linalg.qr(a);                                    // Q, R
Tensor s = a.matmul(a.transpose(-2, -1));                      // symmetric
Tensor[] eig = Linalg.eigh(s);                                 // eigenvalues ascending, eigenvectors
Tensor[] usv = Linalg.svd(a);                                  // U, S, V
```

`Fft` takes and returns complex tensors as real tensors with a trailing axis of 2. A spectrogram
and its inverse:

```java
Tensor audio = Tensor.of(samples, 4, 16000);                   // [B, L]
Tensor window = Tensor.of(hann400, 400);                       // or null for a rectangular window
Tensor spec = Fft.stft(audio, 400, 160, window, true, false);  // [4, 201, frames, 2]
Tensor power = spec.square().sum(-1, false);                   // |X|^2, [4, 201, frames]
Tensor back = Fft.istft(spec, 400, 160, window, true, false, 16000);
```

## Sparse matrices and graph convolution

A `SparseMatrix` holds the structure; its values are a tensor, so they can be fixed or learned.
For a graph convolutional network, `SparseMatrix.gcn` builds the normalised adjacency from the
edge list, and `GraphConv` layers take it:

```java
SparseMatrix.Valued adj = SparseMatrix.gcn(nodes, src, dst);   // edges src[e] -> dst[e]
Sequential gcn = new Sequential(
        new GraphConv(features, 16, 1).graph(adj), new ReLU(),
        new GraphConv(16, classes, 2).graph(adj));
Adam opt = new Adam(gcn.parameters(), 0.01f);
Tensor loss = F.crossEntropy(gcn.forward(Tensor.of(x, nodes, features)), Tensor.of(labels, nodes));
```

With learned values: `SparseMatrix m = SparseMatrix.fromCoo(rows, cols, r, c)` and
`m.mm(values, dense)`, where `values` is a parameter of `m.nnz()` entries.

## Embedding models

word2vec, paragraph vectors and GloVe train like any other model; `nn.text.Corpus` draws their
examples on the host:

```java
Vocabulary vocab = Vocabulary.build(sentences, 5);              // List<String[]>, minimum count
Corpus corpus = new Corpus(vocab.encode(sentences), vocab.counts(), 11);
Word2Vec w2v = new Word2Vec(vocab.size(), 100, Word2Vec.Mode.SKIP_GRAM, 3);
Adam opt = new Adam(w2v.parameters(), 0.02f);
int k = 5;                                                      // negatives per pair
for (int epoch = 0; epoch < epochs; epoch++) {
    int[][] pairs = corpus.skipGramPairs(5, false);             // [centres, contexts]
    for (int at = 0; at + batch <= pairs[0].length; at += batch) {
        Tensor loss = w2v.loss(Arrays.copyOfRange(pairs[0], at, at + batch),
                Arrays.copyOfRange(pairs[1], at, at + batch), corpus.negatives(batch * k), k);
        loss.backward();
        opt.step();
        loss.item();
        opt.zeroGrad();
    }
}
float[] vectors = w2v.vectors();                                // [vocab, 100]
int similar = Corpus.nearest(vectors, 100, vocab.id("king"));
```

CBOW takes `corpus.cbowSamples(window)` (centre words and their flattened contexts), `Doc2Vec`
takes document ids as well, and `GloVe.loss(rows, cols, counts)` takes batches of
`corpus.cooccurrence(window)`.

## Classical machine learning

The estimators in `org.beehive.jitnn.ml` follow scikit-learn: construct, `fit`, then `predict`.
Features are FP32 tensors [samples, features] and class labels `int[]`:

```java
Tensor x = Tensor.of(features, n, d);
int[] clusters = new KMeans(8, 42).fit(x).predict(x);

RandomForest forest = new RandomForest(RandomForest.Task.CLASSIFICATION, 100, 12, 7).fit(x, labels);
int[] predicted = forest.predict(Tensor.of(testFeatures, m, d));

LogisticRegression logit = new LogisticRegression().fit(x, labels);   // C = 1, L-BFGS
Tensor proba = logit.predictProba(Tensor.of(testFeatures, m, d));      // [m, classes]
```

The others are `KNeighbors`, `Pca`, `LinearRegression` (least squares and ridge), `LinearSvc`,
`NaiveBayes`, `GradientBoosting`, `Tsne` and single decision trees (`RandomForest.tree`).

## FP8

FP8 weights with a per-tensor scale, in a linear layer with an FP32 input:

```java
Tensor[] w8 = Fp8.quantize(weight, DType.FP8_E4M3);             // [FP8 weight, scale]
Tensor y = Fp8.linear(x, null, w8[0], w8[1], bias);             // x W^T + bias, FP32 accumulation
```

An FP8 input goes with its scale (`Fp8.quantize(x, DType.FP8_E4M3)`, then `Fp8.linear(x8[0],
x8[1], ...)`). `Fp8.fakeQuantize(x, type)` rounds through FP8 with a straight-through gradient, to
train for FP8.

## Several devices

`ProcessGroup.run` starts one thread per rank, rank r on `devices[r]`, and `DataParallel` keeps the
replicas in step: each rank trains on its share of the batch, and the gradients are averaged
before every optimiser step.

```java
String[] devices = { "0:0", "0:1" };                            // as tornado --devices lists them
List<Float> last = ProcessGroup.run(devices, (rank, collective) -> {
    Sequential model = buildModel();                             // built on this rank's device
    DataParallel dp = new DataParallel(model, rank, collective).broadcastParameters();
    Adam opt = new Adam(model.parameters(), 1e-3f);
    float value = 0;
    for (int step = 0; step < steps; step++) {
        Tensor loss = F.crossEntropy(model.forward(shardX(rank, step)), shardY(rank, step));
        loss.backward();
        dp.averageGradients();
        opt.step();
        value = loss.item();
        opt.zeroGrad();
    }
    return value;
});
```

The default `HostCollective` moves tensors through host memory, so the ranks can be on any mix of
devices and backends (two GPUs, or CUDA and OpenCL on the same GPU). Tensors belong to the device
they were computed on: do not pass them between ranks except through the collective.
`Engine.onDevice(spec, body)` runs a block on another device from any thread. The ranks run one
after another inside jitNN, so this gives correct data-parallel training but no speed-up yet.

## ONNX models

`jitnn-onnx` runs ONNX files on jitNN's operators:

```java
OnnxSession session = OnnxSession.load(Path.of("resnet50.onnx"));      // or load(path, true): FP16 products
Map<String, Tensor> out = session.run(Map.of("input", Tensor.of(images, 1, 3, 224, 224)));
float[] logits = out.get("output").toFloatArray();
```

`session.inputNames()` and `outputNames()` list the graph's names. Integer inputs (token ids) are
passed as INT32 tensors. The outputs are lazy jitNN tensors: read them, or keep computing with
them. The first `run` compiles the graph; later runs with the same input shapes replay it.

## Backends

`Engine.backend()` is `TORNADO` when a TornadoVM runtime is on the class path, else `REFERENCE`:
the host executes each operator eagerly in double precision. `Engine.with(Backend.REFERENCE, ...)`
runs a block on the reference, which is how the tests compare the two.

## Tests

| Command | What runs |
|---|---|
| `./mvnw test` | reference backend: values, finite differences, PyTorch and ONNX goldens, MLP convergence |
| `./mvnw test -Paccel-tests [-Djitnn.test.device=d:i]` | every operator, PyTorch golden and ONNX golden case on a device, the scikit-learn comparisons, the embedding and data-parallel gates, the MLP gate, optimiser goldens |
| `... -Djitnn.mnist=<dir>` | adds the MNIST gate against PyTorch |
| `... -Djitnn.dist.devices=0:0,1:0` | runs the collective and data-parallel tests on these devices (default: two ranks on the test device) |
| `... -Dtest=BaselineBenchAccelTest -Djitnn.test.profile=true` | kernel benchmarks |
| `... -Dtest=InferenceBenchAccelTest -Djitnn.bench.inference=true` | ResNet-50 and BERT-base inference latency |
