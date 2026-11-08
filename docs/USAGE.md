# Using jitNN

## Tensors

A `Tensor` is an n-dimensional, contiguous, row-major array. Its `DType` is `F32` (the default),
`I32` (labels and indices), `F16` or `BF16` (mixed precision), `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`,
`Q5_K` or `Q6_K` (GGUF quantised blocks, for weights) or `FP8_E4M3`/`FP8_E5M2`.

```java
Tensor x = Tensor.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);   // data, then shape; int[] gives INT32
Tensor z = Tensor.zeros(4, 4);
Tensor u = Tensor.uniform(42, -1, 1, 128, 784);                  // seed, range, shape; also Tensor.normal
```

Random tensors use a counter-based generator (Philox4x32-10), so a seed gives the same values on
every device and on the host. Operators are methods (`add`, `mul`, `exp`, `relu`,
`sum(axis, keepDim)`, `max`, `argmax`, `matmul`, `permute`, `reshape`, `softmax`, `logSoftmax`,
...) and functions in `F` (`linear`, `linearRelu`, `crossEntropy`, `nll`, `accuracy`). Binary
operators broadcast like NumPy. [OPERATORS.md](OPERATORS.md) lists every operator.

## Lazy execution and sync points

On a TornadoVM device, operators are recorded, not run. A **sync point** (`item()`,
`toFloatArray()`, `toIntArray()` or `Engine.sync()`) compiles everything recorded on the thread
into one TornadoVM task graph. The plan is cached under the trace's structure (operators, shapes,
parameters) and replayed without compiling when that structure recurs, as on every training step.

- **Read what you need before the next step.** A replayed plan reuses its buffers. Reading a
  tensor from an earlier step that was never read throws `IllegalStateException` ("data was
  reused"). Values already read to the host stay valid.
- **Keep the step's structure stable.** A new batch size or branch compiles a new plan.
  `-Djitnn.verbose=true` logs each compile.
- **Close models you are done with.** Closing an `OnnxSession`, a `TfModel` or a DJL model (or
  calling `Tensor.release()` on its weights) lets the next model of the same structure reuse its
  plans without compiling. A released tensor cannot be used again.

## Autograd

```java
Tensor w = Tensor.of(values, 10, 784).requiresGrad(true);
F.crossEntropy(F.linear(x, w, null), labels).backward();
Tensor g = w.grad();
```

`backward()` records the gradient operators into the same trace, so forward pass, backward pass
and optimiser step become one compiled graph. `Grad.disabled()` turns recording off in a
try-with-resources block.

## Parameters and modules

A **parameter** lives on the device across steps and moves to the host only when read.
`Tensor.parameter(init)` creates one. `Module` subclasses register their parameters; switch to
evaluation with `model.train(false)`. `Linear(in, out, seed)` follows PyTorch's layout and
initialisation, and `Sequential(...)` fuses a `ReLU` into the `Linear`, `Conv2d` or `BatchNorm2d`
before it. [OPERATORS.md](OPERATORS.md) describes how each module is computed.

### CNN layers and models

`Conv2d(in, out, k, stride, padding, seed)` (NCHW, square kernels, PyTorch initialisation; a full
constructor adds dilation, groups and bias), `BatchNorm2d(channels)` (batch statistics in
training, running statistics in evaluation), `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`,
`Flatten`, `ReLU6`, `Dropout(p, seed)` (a new mask every step without recompiling) and
`Add(main, shortcut, relu)` (a residual connection). Models: `ResNet.resnet18(in, classes, small,
seed)`, `ResNet.resnet50(...)` and `MobileNetV2.mobilenetV2(in, classes, small, dropout, seed)`, in
torchvision layouts; `small` adapts the stem to 28-32 pixel inputs.

### Transformer layers and models

`Embedding(count, dim, seed)` (INT32 indices, optional padding index, deterministic gradient),
`LayerNorm(features)` and `RMSNorm(features, eps)` (over the last axis), and
`MultiheadAttention(dim, heads, bias, seed)` (self-attention on [B, T, D],
`forward(x, mask, causal)`). Models in `nn.models`: `GPT2`, `Llama` (RMSNorm, RoPE, SwiGLU,
grouped-query attention) and `Bert` (encoder with MLM head). Functional forms in `F`: `embedding`,
`layerNorm`, `rmsNorm`, `scaledDotProductAttention` (causal and additive masks),
`multiHeadAttention`, `rope`, `swiglu`, `multinomial`. On `Tensor`: `gelu`, `geluTanh`, `silu`,
`sigmoid`, `tanh`, `narrow`, `slice`, `split`, `chunk`, `cat`, `stack`, `topk`.

### Generative and recurrent layers and models

`ConvTranspose2d(in, out, kernel, stride, padding, outputPadding, dilation, groups, bias, seed)`
(PyTorch weight layout [in, out/groups, k, k]), `Upsample(scale, bilinear, alignCorners)`
(`F.interpolate` resizes to a given size), `LeakyReLU(slope)`, `Sigmoid`, `Tanh`, and
`LSTM(input, hidden, layers, seed)` and `GRU(...)` (time-first [T, B, I]; `forward(x, h0, c0)`
returns the sequence and final states). Models in `nn.models`: `Vae`, `Dcgan.Generator`,
`Dcgan.Discriminator`, `UNet`, `CharRnn`. Losses in `F`: `mseLoss`, `binaryCrossEntropy`,
`binaryCrossEntropyWithLogits` (each with a `Reduction`), `gaussianKl`. Noise drawn every step
should take its seed as a tensor, so the step is replayed rather than recompiled:

```java
Tensor eps = Tensor.uniform(Tensor.seed(step), -1.7320508f, 1.7320508f, batch, latent);
```

## Optimisers

`new SGD(params, lr)` or `new SGD(params, lr, momentum, weightDecay, nesterov)`;
`new Adam(params, lr)` or its full constructor; `Adam.adamW(params, lr, weightDecay)`;
`new Lamb(params, lr, beta1, beta2, eps, weightDecay)`;
`new Lars(params, lr, momentum, weightDecay, eta, eps)`; and AdaGrad, RMSProp, AdaDelta, AdaMax and
NAdam in `nn.optim`, with PyTorch's arguments.

Updates run in place on the device; the learning rate and Adam's bias corrections are uploaded per
step, so `setLearningRate` and schedules never recompile. `LRScheduler.step`, `multiStep` and
`cosine` adjust once per epoch; `LRScheduler.warmupLinear` and `warmupCosine` count optimiser steps
(call `step()` after every `opt.step()`). `opt.clipGradNorm(maxNorm)` clips by global norm inside
the next update (PyTorch's `clip_grad_norm_`); `Grad.clipNorm(params, maxNorm)` rescales the
gradient tensors themselves.

## The training loop

```java
for (int step = 0; step < steps; step++) {
    Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, batch, 784)), Tensor.of(yb, batch));
    loss.backward();
    opt.step();
    float value = loss.item();     // one compiled plan, replayed from the second step
    opt.zeroGrad();
}
```

## Activation checkpointing

`Grad.checkpoint(segment, inputs...)` keeps none of a segment's activations; the backward pass runs
the segment again, like `torch.utils.checkpoint`. Intermediates are freed after their last forward
reader, so a step holds one segment's activations at a time. The transformer models have a switch
for their blocks:

```java
GPT2 model = new GPT2(vocab, context, dim, heads, layers, seed).checkpointing(true);
Tensor h = Grad.checkpoint(in -> block.forward(in[0]), x);     // or any segment
```

Gradients reach the segment's inputs and the parameters it uses, and equal those without
checkpointing (bit for bit on the reference backend, within 1e-5 on CUDA). Dropout draws the same
masks on the rerun; BatchNorm in training mode updates its running statistics twice, as in
PyTorch. `Engine.lastIntermediateBytes()` reports the intermediate memory of the last plan.

## Checkpoints

`stateDict()` on a module, optimiser, `GradScaler` or `LRScheduler` returns a `StateDict` of named
tensors and values referring to the live objects. Combine them under prefixes and save one
safetensors file. Loading copies in place, so compiled plans keep running on the loaded values.

```java
new StateDict().put("model", model.stateDict()).put("optimizer", opt.stateDict())
        .put("epoch", epoch).save(Path.of("run.safetensors"));
StateDict in = StateDict.load(Path.of("run.safetensors"));
model.loadStateDict(in.sub("model"));
opt.loadStateDict(in.sub("optimizer"));
int epoch = in.intValue("epoch");
```

Model tensors are named by position (`parameters.<i>`, `buffers.<i>`), so load into a model built
the same way; names, dtypes and shapes are checked first. The optimiser part holds its type,
learning rate, step count and state (`state.<slot>.<i>`, e.g. Adam's `m` and `v`); build it with
the same hyper-parameters. Resuming reproduces an uninterrupted run bit for bit on the same
device. Files are standard safetensors, and `StateDict.load(...).tensor(name)` reads a tensor from
any safetensors file. Types: F32, F16, BF16, I32, both FP8 formats; GGUF blocks cannot be saved.

## Hugging Face models and tokenizers

`GPT2.fromHuggingFace`, `Bert.fromHuggingFace` and `Llama.fromHuggingFace` build a model from a
Hugging Face directory (`config.json` and `model.safetensors`, or sharded safetensors with their
index), as `huggingface-cli download` or `snapshot_download` leave it. `Tokenizer.load` reads its
`tokenizer.json`.

```java
Bert bert = Bert.fromHuggingFace(dir);
Tokenizer.Batch batch = Tokenizer.load(dir).encodeBatch(List.of("The capital of France is [MASK]."), 128);
Tensor hidden = bert.encode(batch.ids(), batch.typeIds(), batch.additiveMask());
Tensor pooled = bert.pooled(hidden);
```

GPT-2 loads from `GPT2Model` or `GPT2LMHeadModel`. `Bert.fromHuggingFace` reads `model_type` and
loads BERT, RoBERTa and DistilBERT (`BertModel`, `BertForMaskedLM`, `RobertaForMaskedLM`,
`DistilBertForMaskedLM` and the like), with the MLM head and pooler when present; RoBERTa's
positions skip padding. Task heads such as a classifier are read with `HfCheckpoint` (`load`,
`floats`, `unused()`). `Llama.fromHuggingFace` takes `model_type` llama, mistral (without a sliding
window) and qwen2 (with q, k, v biases), grouped-query attention, tied or separate output weights,
and linear or Llama 3.1 RoPE scaling (`rope_scaling`, or transformers 5's `rope_parameters`;
`F.ropeHeads` and `Llama.ropeScaling` take a `Rope.Scaling`); for generation with a KV cache, see
[Text generation](#text-generation). Tokenizers: WordPiece with the BERT normaliser, and byte-level
BPE (GPT-2, RoBERTa, and with the `Sequence`, `Digits` and regex `Split` pre-tokenizers and NFC/NFKC
normalisers, SmolLM2 and Qwen2.5); Unigram and SentencePiece-style BPE are rejected. PyTorch `.bin`
checkpoints are not read.

## Exporting to ONNX

`OnnxExport` (in `jitnn-onnx`) writes a forward pass as an ONNX model (opset 20). The pass is
recorded once on example inputs without running; parameters become initializers and the example
inputs the graph inputs.

```java
OnnxExport.export(Path.of("resnet.onnx"), model::forward, Tensor.uniform(1, -1, 1, 1, 3, 224, 224));
OnnxExport.export(Path.of("bert.onnx"), x -> bert.encode(x[0], x[1], x[2]),   // several inputs
        new Tensor[] { ids, types, mask }, new String[] { "input_ids", "token_type_ids", "mask" }, "hidden");
```

Call `model.train(false)` first. Shapes are fixed to the example inputs (no dynamic axes). Weights
are FP32, and protobuf limits files to 2 GB (no external data). Every operator of the `jitnn-nn`
models is mapped; attention becomes `MatMul`/`Softmax` and RMSNorm its arithmetic, and an unmapped
operator is reported by name. The forward pass must not read tensors (`item()`, `toFloatArray()`),
as nothing runs while it is recorded. `scripts/onnx/check_export.py` checks exported files with
ONNX Runtime and `onnx.checker`.

## Data loading

`DataLoader` batches a `Dataset` (`Mnist`, `Cifar10`, or arrays through
`Dataset.of(x, y, shape...)`) like PyTorch's `DataLoader`:

```java
Transforms.Transform aug = Transforms.randomCrop(4).then(Transforms.randomHorizontalFlip())
        .then(Transforms.normalize(mean, std));
try (DataLoader loader = DataLoader.builder(Cifar10.load(dir, true), 128).shuffle(42).transform(aug).workers(4).build()) {
    for (DataLoader.Batch b : loader.epoch(epoch)) { Tensor loss = F.crossEntropy(model.forward(b.x()), b.y()); ... }
}
```

Each epoch is reshuffled from the seed and epoch number; `dropLast(true)` drops a short last
batch. Transforms (`randomCrop(pad)`, `randomHorizontalFlip()`, `normalize(mean, std)` or your own
`Transform`) run per sample on the host. With `workers(n)`, threads assemble the next `prefetch`
batches (default 2) while the device works. Batches come out in order and are identical for any
number of workers, as random streams derive from the seed, epoch and sample.

## Mixed precision

`Tensor.half()`, `bfloat16()` and `to(DType)` convert between FP32, FP16 and BF16 (round to nearest
even). Inside an autocast scope the GEMMs (`F.linear`, `matmul`, and attention's two products when
not fused) take half-precision operands with FP32 accumulation; results and all other operators
stay FP32. In FP16 inference, linear weights are cast once per weight version and cached. FP16
needs dynamic loss scaling with a `GradScaler`:

```java
try (var ac = Autocast.enable(DType.F16)) {        // or DType.BF16, usually without a scaler
    loss = F.crossEntropy(model.forward(x), y);
}
scaler.scale(loss).backward();                     // GradScaler scaler = new GradScaler()
scaler.unscale(opt);                               // before clipping
opt.clipGradNorm(1.0f);
scaler.step(opt);                                  // skipped on the device if a gradient overflowed
loss.item();
scaler.update();                                   // reads the overflow flag, adjusts the scale
```

The scale is uploaded with each step and overflow is detected on the device, so the step stays one
plan. BF16 on OpenCL and Metal needs a TornadoVM with float-bit reinterpretation (TornadoVM#1141).

### FP16 training

`Autocast.enableHalf()` keeps activations in FP16 between operators that have an FP16 form (linear
layers, element-wise operators, LayerNorm, RMSNorm, attention, and convolutions with
`-Djitnn.native`). Weights stay FP32 as the optimiser's master copy and are cast inside the graph.
Use it in place of `Autocast.enable` in the loop above. The backward pass runs the same way
whatever scope `backward()` is called in. Parameters get FP32 gradients; activation gradients are
mostly FP32. On CUDA, attention with FP16 inputs runs forward and backward on tensor cores (head
dimensions up to 128; `-Djitnn.flash.mma=false` turns it off).

### FP16 inference

With gradients off, `Autocast.enableHalf()` runs inference in half precision end to end (with
gradients on, it behaves as `Autocast.enable(DType.F16)`). Linear layers, element-wise operators,
LayerNorm, RMSNorm and attention take and return FP16 and compute in FP32 inside; other operators
take FP32 copies and return FP32. Weights are cast to FP16 once per version. With
`-Djitnn.native`, convolutions also run in FP16 (cuDNN, channels-last, on CUDA; MLX on Metal),
products go to cuBLASLt with bias and activation in the epilogue, and attention to cuDNN's fused
kernel, CUTLASS or MLX; see [NATIVE.md](NATIVE.md).

```java
Tensor logits = Grad.inference(() -> { try (var h = Autocast.enableHalf()) { return model.forward(ids); } });
float[] out = logits.toFloatArray();                  // FP16 results read as floats
```

## Kernels and memory traffic

`Engine.trafficStats()` returns running totals `[executions, kernel launches, library calls,
bytes]`, the bytes counting each operator's inputs and outputs once. Subtract two calls around a
step to get its counts, for example to see what fusion (`-Djitnn.fuse`) saves.

### Timeline

With `-Djitnn.profile=true` (and `-Djitnn.cudaGraph=false` on CUDA), `Engine.startTrace()` and
`Engine.stopTrace(file)` record Chrome trace events for [Perfetto](https://ui.perfetto.dev) or
`chrome://tracing`:

```java
Engine.startTrace();
try (AutoCloseable r = Engine.traceRange("step " + step)) { ... }
Engine.stopTrace(Path.of("trace.json"));
```

Tracks show `Engine.traceRange` spans, host plan runs (first runs, which compile, are marked) and
device kernels named after their operators. TornadoVM reports durations, not start times, so
kernels are drawn back to back in launch order from the start of their plan's run.

### Determinism

`Engine.setDeterministic(true)` (or `-Djitnn.deterministic=true`) makes runs reproducible bit for
bit on the same device, library versions and inputs, like PyTorch's `use_deterministic_algorithms`.
jitNN's own kernels are already deterministic (no atomics, fixed reduction order). The switch also
stops autotuning from measuring and asks cuDNN for deterministic algorithms only; set it before the
first convolution, as cuDNN plans are cached.

## Autotuning

jitNN picks a kernel variant for each operator by built-in rules. With `-Djitnn.autotune=on` it
measures instead: the first time a shape runs, each variant of its GEMMs, 1x1 and 3x3 convolutions
and row norms is timed on random inputs, checked against the rule's result, and kept only if more
than 5% faster than the rule's choice. Decisions are saved per device in `~/.jitnn/autotune`, so
the cost (mostly compiling the variants) is paid once. `-Djitnn.autotune=use` applies saved
decisions without measuring. `Autotune.setMode` and `Autotune.setDirectory` do the same from code.

## Serving

**Concurrent requests.** `-Djitnn.concurrency=N` lets up to N threads run plans on the device at
once. Each slot has its own plans and device buffers, weights included, so memory grows with N;
threads in the same slot take turns. A plan that writes persistent tensors (parameters, optimiser
state) runs alongside plans that do not touch them; plans that need them wait.

**Release outputs once read.** Before a plan runs again, jitNN copies its earlier outputs into host
arrays of their own, so they stay valid. Releasing an output once read (`Tensor.release`, or
try-with-resources) skips that copy, and released or collected outputs' host arrays are reused (up
to `-Djitnn.hostPool.bytes`). Reading a released tensor is an error.

```java
try (Tensor logits = Grad.inference(() -> model.forward(x))) { float[] out = logits.toFloatArray(); ... }
```

**Request shapes.** Kernels are specialised on their sizes, so each new batch size or sequence
length compiles a plan. `OnnxSession.withBuckets` pads requests to a few shapes instead:

```java
OnnxSession session = OnnxSession.load(Path.of("encoder.onnx")).withBuckets(1, 2, 4, 8, 16, 32, 64, 128, 256, 512);
Map<String, Tensor> out = session.run(Map.of("input_ids", ids, "attention_mask", mask));   // any [batch, length]
```

Symbolic dimensions round up to the smallest bucket that holds them (larger ones run as they are),
but only safe ones: the first axis of every input it appears in, and dimensions of inputs named
`*mask*`. Outputs are cut back, so results equal unpadded runs. Each bucket compiles on first use,
so warm each up at start-up; `withBuckets` reserves plan-cache room. `-Djitnn.onnx.buckets=1,2,4,...`
gives every ONNX session buckets, DJL's and TensorFlow's included.

**Buckets for jitNN models.** `Buckets` holds the sizes a dimension rounds up to; `Bucketing` runs a
model on padded inputs and returns only the real rows and positions:

```java
Buckets batch = Buckets.powersOfTwo(1, 16), seq = Buckets.multiplesOf(64, 256);
Engine.reservePlans(20);   // 5 batch x 4 sequence sizes
Tensor logits = Grad.inference(() -> Bucketing.causal(batch, seq, idx, 0, gpt2::forward));      // GPT-2, Llama
Tensor hidden = Grad.inference(() -> Bucketing.masked(batch, seq, idx, mask, 0, bert::encode)); // BERT
Tensor scores = Grad.inference(() -> Bucketing.rows(batch, images, resnet::forward));           // CNNs
Tensor loss = Bucketing.crossEntropy(model.forward(Buckets.pad(x, 0, batch.size(n), 0)), labels);  // training
```

Padding happens on the host. Causal models pad sequences at the end; `masked` gives padded
positions `-1e9`. `Bucketing.crossEntropy` weighs real rows 1/N and padded rows 0, so loss and
gradients equal the real rows' and every N in a bucket shares a plan. Do not bucket BatchNorm in
training mode (padded rows would enter its statistics). Make inputs on the host (`Tensor.of`):
`Tensor.uniform(seed, ...)` puts the seed in its plan. The default plan cache holds 16 plans.
Padded rows cost device time, so choose closer buckets when shapes repeat often.

## Inference and memory

Inside `Grad.inference(() -> model.forward(x))` no gradients are recorded, and every intermediate
but the returned tensor gives its buffer back once no longer needed. Gradient temporaries in
`backward()` are reused the same way; forward activations of a training step are kept, so they can
still be read. In evaluation mode and inside `Grad.inference`, convolution + BatchNorm (+ ReLU)
runs as one convolution on folded weights, a GELU after a linear layer runs in the GEMM epilogue,
and add + ReLU is one kernel (see [OPERATORS.md](OPERATORS.md#inference-fusion)).

## More losses

`F` also has `ctcLoss`, `tripletMarginLoss`, `contrastiveLoss`, `cosineEmbeddingLoss`,
`marginRankingLoss`, PyTorch's regression and classification losses (L1, Huber, smooth L1, hinge,
Poisson NLL, ...), `sparsemax` and `logSigmoid`:

```java
// logProbs [T, B, C] after logSoftmax over C; targets [B, maxLabels] INT32, padded
Tensor loss = F.ctcLoss(logProbs, targets, inputLengths, targetLengths, 0, F.Reduction.MEAN);
```

## Linear algebra and Fourier transforms

`Linalg` works on a matrix or a batch [..., rows, cols], and every function has a gradient:
`solve` (A X = B), `slogdet` ([sign, log|det|]), `qr` (Q, R), `eigh` (symmetric input; eigenvalues
ascending, eigenvectors) and `svd` (U, S, V).

`Fft` represents complex tensors as real tensors with a trailing axis of 2:

```java
Tensor x = Linalg.solve(a, b);                                 // a [64, 8, 8], b [64, 8, 3] -> [64, 8, 3]
Tensor[] usv = Linalg.svd(a);
Tensor spec = Fft.stft(audio, 400, 160, window, true, false);  // audio [4, 16000] -> [4, 201, frames, 2]; window may be null
Tensor back = Fft.istft(spec, 400, 160, window, true, false, 16000);
```

## Sparse matrices and graph convolution

A `SparseMatrix` holds the structure; its values are a tensor, fixed or learned.
`SparseMatrix.gcn` builds a graph convolutional network's normalised adjacency from an edge list,
and `GraphConv` layers take it:

```java
SparseMatrix.Valued adj = SparseMatrix.gcn(nodes, src, dst);   // edges src[e] -> dst[e]
Sequential gcn = new Sequential(new GraphConv(features, 16, 1).graph(adj), new ReLU(),
        new GraphConv(16, classes, 2).graph(adj));
```

With learned values: `SparseMatrix m = SparseMatrix.fromCoo(rows, cols, r, c)` and
`m.mm(values, dense)`, where `values` is a parameter of `m.nnz()` entries.

## Embedding models

word2vec, paragraph vectors and GloVe train like any model; `nn.text.Corpus` draws their examples on
the host:

```java
Vocabulary vocab = Vocabulary.build(sentences, 5);              // List<String[]>, minimum count
Corpus corpus = new Corpus(vocab.encode(sentences), vocab.counts(), 11);
Word2Vec w2v = new Word2Vec(vocab.size(), 100, Word2Vec.Mode.SKIP_GRAM, 3);
int[][] pairs = corpus.skipGramPairs(5, false);                 // [centres, contexts], once per epoch
Tensor loss = w2v.loss(centres, contexts, corpus.negatives(batch * k), k);   // a batch of pairs, k negatives each
float[] vectors = w2v.vectors();                                // [vocab, 100]
int similar = Corpus.nearest(vectors, 100, vocab.id("king"));
```

CBOW takes `corpus.cbowSamples(window)`, `Doc2Vec` also takes document ids, and
`GloVe.loss(rows, cols, counts)` takes batches of `corpus.cooccurrence(window)`.

## Classical machine learning

The estimators in `ai.beestream.jitnn.ml` follow scikit-learn: construct, `fit`, `predict`.
Features are FP32 tensors [samples, features]; class labels are `int[]`:

```java
int[] clusters = new KMeans(8, 42).fit(x).predict(x);
int[] predicted = new RandomForest(RandomForest.Task.CLASSIFICATION, 100, 12, 7).fit(x, labels).predict(test);
Tensor proba = new LogisticRegression().fit(x, labels).predictProba(test);   // C = 1, L-BFGS; [m, classes]
```

Also available: `KNeighbors`, `Pca`, `LinearRegression` (least squares and ridge), `LinearSvc`,
`NaiveBayes`, `GradientBoosting`, `Tsne` and single decision trees (`RandomForest.tree`).

## Quantised weights from GGUF files

`Tensor.quantized` wraps the raw GGUF blocks of a weight [out, in], and `Gguf.linear` multiplies by
it without expanding it (jitllm's kernels for up to 32 input rows, a dequantised copy beyond). The
gradient reaches `x`, not the weight. Q8_0 uses the same call; `Gguf.dequantize(w)` gives FP32.
`jitnn-gguf` reads files: `GgufFile.open` reads metadata and the tensor list, and `tensor(name)`
returns a persistent tensor (block formats as blocks; F32, F16, BF16 as such; shapes [out, in],
ready for `Gguf.linear`). Tensors are memory-mapped copy-on-write, so loading uses no Java heap and
never changes the file; a file that cannot be opened for writing is copied (`tensor(name, true)`
asks for a copy).

```java
Tensor w = Tensor.quantized(DType.Q4_K, blocks, 4096, 4096);    // raw bytes of a Q4_K tensor
GgufFile f = GgufFile.open(Path.of("Llama-3.2-1B-Instruct-Q4_K_M.gguf"));
int layers = f.integer("llama.block_count");                     // also floatValue, strings
Tensor y = Gguf.linear(x, f.tensor("blk.0.attn_q.weight"), null);
```

Types jitNN does not hold (Q8_1, Q8_K, the IQ formats) are listed and `raw(name)` returns their
bytes, but they do not load as tensors.

### Text generation

The optional module `jitnn-llm` (built with `-Pllm`) runs a GGUF language model through jitllm,
which provides tokenizers, a KV cache and kernels for Llama, Mistral, Qwen, Phi, Granite, Gemma and
other families:

```java
try (Llm llm = Llm.load(Path.of("Llama-3.2-1B-Instruct-Q8_0.gguf"))) {
    String answer = llm.generate("What is the capital of France?", 64);   // greedy
    llm.generate("Write a haiku about GPUs.", 64, System.out::print);      // streamed
    Tensor wq = llm.gguf().tensor("blk.0.attn_q.weight");                  // the same file's weights
}
```

`generate(GenerationRequest)` takes jitllm's requests (sampling, stop sequences, chat messages,
tools); `model()` returns the jitllm model. jitllm runs on TornadoVM's default device, as jitNN does
without `-Djitnn.device`; backend priorities choose it (e.g. `-Dtornado.cuda.priority=100`).

## FP8

FP8 weights with a per-tensor scale, in a linear layer with an FP32 input:

```java
Tensor[] w8 = Fp8.quantize(weight, DType.FP8_E4M3);             // [FP8 weight, scale]
Tensor y = Fp8.linear(x, null, w8[0], w8[1], bias);             // x W^T + bias, FP32 accumulation
```

An FP8 input goes with its scale (`Fp8.quantize(x, DType.FP8_E4M3)`, then
`Fp8.linear(x8[0], x8[1], ...)`). `Fp8.fakeQuantize(x, type)` rounds through FP8 with a
straight-through gradient, to train for FP8.

## Several devices

`ProcessGroup.run` starts one thread per rank, rank r on `devices[r]` (as `tornado --devices` lists
them). `DataParallel` keeps the replicas in step: each rank trains on its share of the batch and
gradients are averaged before each optimiser step. Build the model inside the rank's body, so it
lives on that rank's device.

```java
List<Float> last = ProcessGroup.run(new String[] { "0:0", "0:1" }, (rank, collective) -> {
    Sequential model = buildModel();
    DataParallel dp = new DataParallel(model, rank, collective).broadcastParameters();
    ...                                                          // forward, loss.backward()
    dp.averageGradients();
    opt.step();
    ...
});
```

The default `HostCollective` moves tensors through host memory, so ranks can use any mix of devices
and backends, but each collective waits for the host. On NVIDIA GPUs, `NcclCollective` keeps
collectives on the GPUs: they are recorded into each rank's plan (and captured in its CUDA graph),
and `averageGradients` reduces all gradients in one packed NCCL call, so a step is one plan per rank.
With NCCL, results are new tensors (`allReduced`) or persistent tensors written in place
(parameters, buffers, `Tensor.persistent`).

```java
try (NcclCollective nccl = new NcclCollective(devices)) {        // distinct CUDA GPUs
    ProcessGroup.run(devices, nccl, (rank, collective) -> train(rank, collective));
    Engine.reset();                                              // release the plans before the communicator
}
```

`dp.overlapGradients(bucketBytes)` overlaps the all-reduce with the backward pass, like PyTorch's
DistributedDataParallel: parameters are grouped, last layer first, into buckets of about
`bucketBytes`, each reduced on its own CUDA stream once its last gradient is final, and
`averageGradients` only waits. Results are bit-identical to no overlap. Under CUDA graphs an
all-reduce is waited for at the latest at the end of the TornadoVM task graph that started it;
`-Djitnn.graph.maxTasks` (with a larger `-Dtornado.tvm.maxbytecodesize`) gives longer graphs. Ranks
in one process need a TornadoVM whose CUDA backend unpins host memory off the plan threads
(TornadoVM#1289). Other collectives reduce in `averageGradients`, which assumes one backward pass
per step. `SyncBatchNorm2d(channels, collective, rank)` replaces `BatchNorm2d` with whole-batch
statistics across ranks (one differentiable NCCL all-reduce per layer).

Ranks may span processes and machines: one process calls `NcclCollective.uniqueId()` and passes
the id on; each process creates `new NcclCollective(id, worldSize, firstRank, "0:0")` and calls
`ProcessGroup.run` for its own ranks. Tensors belong to the device they were computed on, so pass
them between ranks only through the collective. `Engine.onDevice(spec, body)` runs a block on
another device from any thread.

## ONNX models

`jitnn-onnx` runs ONNX files on jitNN's operators:

```java
OnnxSession session = OnnxSession.load(Path.of("resnet50.onnx"));      // or load(path, true): FP16 products
float[] logits = session.run(Map.of("input", images)).get("output").toFloatArray();
```

`session.inputNames()` and `outputNames()` list the graph's names. Integer inputs (token ids) are
INT32 tensors. Outputs are lazy jitNN tensors. The first `run` compiles the graph; later runs with
the same input shapes replay it.

## TensorFlow models

`jitnn-tf` runs TensorFlow SavedModels and frozen GraphDefs:

```java
TfModel model = TfModel.load(Path.of("exported_model"));              // a SavedModel directory, or a .pb file
Map<String, Tensor> out = model.run(Map.of("image", Tensor.of(pixels, 1, 224, 224, 3)));
```

Inputs and outputs carry the signature's names (`model.inputNames()`, `outputNames()`). Images
stay NHWC; integer inputs are INT32 tensors.

## DJL

With `jitnn-djl` on the class path, DJL code runs on jitNN unchanged:

```java
Model model = Model.newInstance("mlp");                            // jitNN is the default engine
model.setBlock(block);
Trainer trainer = model.newTrainer(config);
trainer.initialize(new Shape(32, 784));
EasyTrain.fit(trainer, epochs, trainSet, validateSet);
```

The engine is named `jitNN` with rank 0, so it is DJL's default; `-Dai.djl.default_engine=PyTorch`
or naming an engine in `Model.newInstance` picks another. DJL operators are recorded as jitNN
traces, compiled at the first sync point and replayed on later steps of the same shape.
`Model.load` reads ONNX files. NDArrays follow DJL's types: integer division truncates, and an
integer array with a floating scalar gives FLOAT32, as in PyTorch. Operators jitNN lacks run on the
engine named by `-Djitnn.djl.fallback`, when it is on the class path. DJL's examples (`TrainMnist`,
`TrainResnetWithCifar10`, `ImageClassification`, `YoloDetection`) run unchanged; performance
results are in [jitNN-benchmarking](https://github.com/beestream-ai/jitNN-benchmarking).

## Custom operators

A custom operator is a kernel in plain Java using TornadoVM's `KernelContext`, plus an optional host
reference and gradient. It behaves like a built-in operator: recorded lazily, compiled into the same
plan as its neighbours, cached and differentiated.

```java
public static void swish(KernelContext kc, FloatArray x, FloatArray y, int n) {
    int i = kc.globalIdx;
    if (i < n) {
        float v = x.get(i);
        y.set(i, v / (1.0f + TornadoMath.exp(-v)));
    }
}

static final CustomOp SWISH = CustomOp.unary("swish", MyKernels::swish, v -> v / (1 + Math.exp(-v)))
        .withGradient((x, y, gy) -> gy.mul(y.add(x.sigmoid().mul(y.neg().add(1)))));
Tensor y = SWISH.apply(x);
```

`CustomOp.unary` and `CustomOp.binary` cover element-wise FP32 operators: the kernel runs on a 1-D
grid of at least `n` threads and bound-checks its index. `CustomOp.builder(name)` covers the rest:
input types, one or more outputs with shape functions, the tasks to launch (each with an explicit
grid from `Emitter`), the host reference and the gradient.

```java
static final CustomOp MIN_MAX = CustomOp.builder("minMax")
        .inputs(DType.F32)
        .output((in, attrs) -> Shape.of(in[0].shape().dim(0)), DType.F32)
        .output((in, attrs) -> Shape.of(in[0].shape().dim(0)), DType.F32)
        .launch((e, n) -> {
            FloatArray x = e.f32(n.in(0)), min = e.f32(n.out(0)), max = e.f32(n.out(1));
            int rows = n.in(0).shape().dim(0), cols = n.in(0).shape().dim(1);
            e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, MyKernels::minMax, e.context(), x, min, max, rows, cols));
        })
        .reference(n -> { /* the same on the host, into n.out(0) and n.out(1) */ })
        .build();
Tensor[] mm = MIN_MAX.applyAll(null, null, x);
```

`apply(iattr, fattr, inputs...)` passes integer and float attributes, which the launcher reads from
`n.iattr` and `n.fattr`. Plans are cached by operator name and attributes, so names are global:
building a second operator with a used name throws. Without a reference the operator runs only on
a device; without a gradient, `backward()` through it throws. TornadoVM names the generated kernel
after the Java method, so a kernel method must not be named like a CUDA or OpenCL built-in (`exp`,
`hypot`, `max`, ...) or a C++ keyword (`not`, `and`, ...); `unary` and `binary` reject such names.

## Printing the kernels

`Engine.printKernels(System.out)` prints every plan compiled from then on: its operators with
shapes and task numbers, then the CUDA, OpenCL or Metal source of each kernel, custom ones
included. `Engine.printKernels(null)` stops it; `-Djitnn.printKernels=true` (standard output) or
`-Djitnn.printKernels=<file>` turns it on at start-up. Compiled plans are not printed again;
`Engine.reset()` drops them.

```
// ==== jitNN plan trace on CUDA NVIDIA A100 80GB PCIe: 2 operators, 2 kernels in 1 graph(s)
//   [0] custom:swish [77, 3] F32 -> [77, 3] F32  tasks t0
//   [1] binary [0] [77, 3] F32, [] F32 -> [77, 3] F32  tasks t1
extern "C" __global__ void swish(...)
```

## Vendor libraries

By default every operator runs on jitNN's own kernels. With `-Djitnn.native=auto`, matrix
products, convolutions, softmax, inference
attention, FFTs and sparse products call cuBLAS, cuDNN, cuFFT and cuSPARSE on CUDA, or MLX on
Metal, as library tasks inside the same compiled plans. `-Djitnn.native=cublas,cudnn` enables only
the named libraries. A library that is missing or fails is reported once, and its operators keep
jitNN's kernels. [NATIVE.md](NATIVE.md) lists what each library covers, its numerics (cuDNN
convolutions use TF32 on Ampere unless `NVIDIA_TF32_OVERRIDE=0`) and its requirements.

## Backends

`Engine.backend()` is `TORNADO` when a TornadoVM runtime is on the class path, else `REFERENCE`,
where the host runs each operator eagerly in double precision. `Engine.with(Backend.REFERENCE, ...)`
runs a block on the reference, which is how the tests compare the two.

## Tests

| Command | What runs |
|---|---|
| `./mvnw test` | reference backend: values, finite differences, PyTorch and ONNX goldens, MLP convergence |
| `./mvnw test -Paccel-tests [-Djitnn.test.device=d:i]` | every operator, PyTorch and ONNX golden on a device, scikit-learn comparisons, embedding, data-parallel and MLP gates, optimiser goldens |
| `... -Djitnn.mnist=<dir>` | adds the MNIST gate against PyTorch |
| `... -Djitnn.dist.devices=0:0,1:0` | runs the collective and data-parallel tests on these devices (default: two ranks on the test device) |
| `... -Dtest=BaselineBenchAccelTest -Djitnn.test.profile=true` | kernel benchmarks |
| `... -Dtest=InferenceBenchAccelTest -Djitnn.bench.inference=true` | ResNet-50 and BERT-base inference benchmarks |
| `... -Djitnn.test.extra=-Djitnn.native=auto` | device tests with the vendor libraries, plus `NativeAccelTest` (each library-backed operator against the reference, and the fallback) |
