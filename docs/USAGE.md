# Using jitNN

## Tensors

A `Tensor` is an n-dimensional, contiguous, row-major array. Its `DType` is `F32` (the default),
`I32` (labels and indices), `F16` or `BF16` (mixed precision), `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`,
`Q5_K` or `Q6_K` (GGUF's quantised blocks, for weights) or `FP8_E4M3`/`FP8_E5M2` (8-bit floats).

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
- **Close models you are done with.** Closing an `OnnxSession`, a `TfModel` or a DJL model (or
  calling `Tensor.release()` on its weights) lets the next model of the same structure, such as the
  same file loaded again, reuse the closed model's plans: its weights are copied into them and
  nothing is compiled. A released tensor cannot be used again.

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
| `GPT2`, `Llama`, `Bert` in `nn.models` | a GPT-2 decoder, a Llama-style decoder (RMSNorm, RoPE, SwiGLU, grouped-query attention) and a BERT encoder with an MLM head |

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

## Activation checkpointing

`Grad.checkpoint(segment, inputs...)` keeps none of a segment's activations for the backward pass.
The backward pass runs the segment again to get them, as PyTorch's `torch.utils.checkpoint` does.
The segment's intermediates are freed after their last forward reader, so a step holds the
activations of one segment at a time. The transformer models have a switch for their blocks:

```java
GPT2 model = new GPT2(vocab, context, dim, heads, layers, seed).checkpointing(true);

// or any segment
Tensor h = Grad.checkpoint(in -> block.forward(in[0]), x);
```

- Gradients reach the segment's inputs and the parameters it uses. They are bit for bit the
  same as without checkpointing on the reference backend, and within 1e-5 on CUDA.
- Dropout draws the same masks when the segment runs again. BatchNorm in training mode updates
  its running statistics twice, as in PyTorch.
- On an RTX 5070 Ti, a GPT-2 training step (dim 512, 8 layers, 8 x 256 tokens) uses
  381 MB of intermediates instead of 763 MB. It took 100 ms instead of 111 ms: fewer and smaller
  buffers outweighed the extra forward pass here.
- `Engine.lastIntermediateBytes()` reports the intermediate memory of the last plan that ran.

## Checkpoints

`stateDict()` on a module, an optimiser, a `GradScaler` or an `LRScheduler` returns a `StateDict`:
named tensors and values that refer to the live objects. Combine the parts under prefixes and save
them as one safetensors file. Loading copies each tensor into the existing parameter or optimiser
state in place, so plans already compiled keep running on the loaded values.

```java
new StateDict()
        .put("model", model.stateDict())
        .put("optimizer", opt.stateDict())
        .put("scheduler", scheduler.stateDict())
        .put("epoch", epoch)
        .save(Path.of("run.safetensors"));

StateDict in = StateDict.load(Path.of("run.safetensors"));
model.loadStateDict(in.sub("model"));
opt.loadStateDict(in.sub("optimizer"));
scheduler.loadStateDict(in.sub("scheduler"));
int epoch = in.intValue("epoch");
```

- Model tensors are named by position (`parameters.<i>`, `buffers.<i>`), so a checkpoint loads into a
  model built the same way. Names, dtypes and shapes are all checked before anything is written.
- The optimiser part holds its type, learning rate, step count and per-parameter state
  (`state.<slot>.<i>`, e.g. Adam's `m` and `v`). Build the optimiser with the same hyper-parameters
  before loading.
- Resuming reproduces an uninterrupted run bit for bit on the same device.
- The files are standard safetensors: `safetensors.torch.load_file` reads them in Python.
  `StateDict.load(...).tensor(name)` reads a tensor from any safetensors file as a new weight.
  Supported types are F32, F16, BF16, I32 and the two FP8 formats. GGUF block-quantised weights
  cannot be saved.

## Hugging Face models and tokenizers

`GPT2.fromHuggingFace`, `Bert.fromHuggingFace` and `Llama.fromHuggingFace` build a model from a Hugging Face model directory
(`config.json` and `model.safetensors`, or sharded safetensors with their index), as
`huggingface-cli download` or `snapshot_download` leave it. `Tokenizer.load` reads the
directory's `tokenizer.json`.

```java
Path dir = Path.of("bert-base-uncased");
Bert bert = Bert.fromHuggingFace(dir);
Tokenizer tok = Tokenizer.load(dir);
Tokenizer.Batch batch = tok.encodeBatch(List.of("The capital of France is [MASK].", "Hello world!"), 128);
Tensor hidden = bert.encode(batch.ids(), batch.typeIds(), batch.additiveMask());
Tensor pooled = bert.pooled(hidden);
```

- Supported: GPT-2 checkpoints (`GPT2Model`, `GPT2LMHeadModel`), and BERT, RoBERTa and DistilBERT
  checkpoints through `Bert.fromHuggingFace`, which reads `model_type` (`BertModel`,
  `BertForMaskedLM`, `RobertaForMaskedLM`, `DistilBertForMaskedLM` and the like). The MLM head and
  pooler are loaded when present. RoBERTa's positions skip padding, as in Hugging Face.
  Task heads, such as a classifier, are read with `HfCheckpoint` (`load`, `floats`, `unused()`).
- Llama-family decoders through `Llama.fromHuggingFace`: `model_type` llama, mistral (without a
  sliding window) and qwen2 (with its q, k, v biases), grouped-query attention, tied or separate
  output weights. RoPE scaling (Llama 3.1 and later) is rejected. SmolLM2-135M and Qwen2.5-0.5B
  logits match `transformers` within 3e-6 of the largest value on CUDA. Their tokenizers (a
  `Sequence` pre-tokenizer) are not read yet; for generation with a KV cache use jitllm.
- Tokenizers: WordPiece with the BERT normaliser, and byte-level BPE (GPT-2, RoBERTa). Other
  tokenizer models (Unigram, SentencePiece-style BPE) are rejected when loading.
- On an RTX 5070 Ti, outputs match `transformers` FP32 on the CPU within 2e-6 of the largest value
  (BERT) and 2.3e-6 (GPT-2 logits). Token ids match the `tokenizers` library on the test texts.
- PyTorch `.bin` checkpoints are not read; most models on the Hub also ship safetensors.

## Exporting to ONNX

`OnnxExport` (in `jitnn-onnx`) writes a forward pass as an ONNX model (opset 20) for ONNX Runtime
and other ONNX tools. The forward pass is recorded once on example inputs without running it.
Parameters and other constants become initializers, and the example inputs become the graph
inputs.

```java
model.train(false);
OnnxExport.export(Path.of("resnet.onnx"), model::forward, Tensor.uniform(1, -1, 1, 1, 3, 224, 224));

// several inputs, e.g. BERT with token types and a padding mask
OnnxExport.export(Path.of("bert.onnx"), x -> bert.encode(x[0], x[1], x[2]),
        new Tensor[] { ids, types, mask }, new String[] { "input_ids", "token_type_ids", "mask" }, "hidden");
```

- Shapes are fixed to those of the example inputs; there are no dynamic axes.
- Weights are written in FP32. Files are limited by protobuf to 2 GB (no external data yet).
- Every operator of the models in `jitnn-nn` is mapped, including MLPs, ResNet, MobileNetV2,
  GPT-2, BERT (with masks, pooler and MLM head) and Llama (RoPE). Attention becomes
  `MatMul`/`Softmax`, and RMSNorm its arithmetic. An operator without a mapping is reported by
  name.
- Code in the forward pass must not read tensors (no `item()` or `toFloatArray()`), as nothing
  runs while it is recorded.
- On an RTX 5070 Ti, ONNX Runtime on the exported Hugging Face `bert-base-uncased` and `gpt2`
  agrees with jitNN within 2e-6 of the largest output. `OnnxExportTest` and
  `OnnxExportAccelTest` check every model with jitNN's importer, and `scripts/onnx/check_export.py`
  checks them with ONNX Runtime and `onnx.checker`.

## Data loading

`DataLoader` batches a `Dataset` (`Mnist`, `Cifar10`, or arrays through `Dataset.of(x, y, shape...)`)
as PyTorch's `DataLoader` does:

```java
Transforms.Transform aug = Transforms.randomCrop(4).then(Transforms.randomHorizontalFlip())
        .then(Transforms.normalize(new float[] { 0.49f, 0.48f, 0.45f }, new float[] { 0.25f, 0.24f, 0.26f }));
try (DataLoader loader = DataLoader.builder(Cifar10.load(dir, true), 128).shuffle(42).transform(aug).workers(4).build()) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (DataLoader.Batch b : loader.epoch(epoch)) {
            Tensor loss = F.crossEntropy(model.forward(b.x()), b.y());
            ...
        }
    }
}
```

- Every epoch is reshuffled from the seed and the epoch number. `dropLast(true)` leaves out a
  short last batch.
- Transforms run per sample on the host while the batch is assembled: `randomCrop(pad)`,
  `randomHorizontalFlip()`, `normalize(mean, std)`, or your own `Transform`.
- With `workers(n)`, threads assemble the next `prefetch` batches (default 2) while the device
  works on the current one.
- Batches come out in order and are the same for any number of workers. Shuffling and transforms
  draw from random streams derived from the seed, the epoch and the sample, so a run is
  reproducible.

## Mixed precision

`Tensor.half()`, `bfloat16()` and `to(DType)` convert between FP32, FP16 and BF16 (round to
nearest even, as PyTorch). Inside an autocast scope the GEMMs (`F.linear`, `matmul`, and the two
products of attention when it is not fused) run on half-precision operands with FP32
accumulation; their results and every other operator stay FP32, including the fused attention
kernels and the norms. In inference under FP16 autocast, linear weights are cast to FP16 once per
version of the weight and cached. FP16 needs dynamic loss scaling, as with PyTorch's `GradScaler`:

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

### FP16 training

`Autocast.enableHalf()` also works in training: activations stay FP16 between the operators that
have an FP16 form (linear layers, element-wise operators, LayerNorm and RMSNorm, attention, and the
convolutions with `-Djitnn.native`). The weights stay FP32, as the master copy the optimiser
updates, and are cast to FP16 inside the graph.

```java
try (var half = Autocast.enableHalf()) {
    loss = F.crossEntropy(model.forward(x), y);
}
scaler.scale(loss).backward();     // runs with FP16 activations too
scaler.step(opt);
```

- The backward pass runs the same way, whatever scope `backward()` is called in.
- Parameters get FP32 gradients. The activations' gradients stay as the operators compute them,
  mostly FP32.
- Losses follow FP32 training closely: within 5e-4 for GPT-2 and BERT over 12 steps, and 1e-2
  for ResNet-18.
- On CUDA, attention with FP16 inputs runs forward and backward on tensor cores (head dimensions up
  to 128; `-Djitnn.flash.mma=false` turns it off). On an A100 the GPT-2 step below spends 2.7 ms in
  attention instead of 4.7 ms at head dimension 64, 2.5 ms instead of 7.7 ms at 32, and 4.1 ms
  instead of 23.8 ms at 128.
- On an RTX 5070 Ti, a GPT-2 step (768 wide, 12 layers, 16 x 128 tokens) uses 13% less memory
  for intermediates than FP16 autocast, at the same speed. It is dominated by the matrix products,
  which autocast already runs in FP16.

### FP16 inference

`Autocast.enableHalf()` runs inference in half precision end to end, like a model converted to FP16.

- **FP16 operators.** Linear layers, element-wise operators, LayerNorm and RMSNorm, and attention
  take and return FP16 tensors and compute in FP32 inside. Activations therefore stay FP16 between
  them, with half the memory traffic of autocast.
- **Other operators** take FP32 copies of FP16 inputs and return FP32.
- **Weights** are cast to FP16 once per version, as with `Autocast.enable`.
- **With `-Djitnn.native`**, convolutions run in FP16 too: on cuDNN's tensor cores, channels-last, on
  CUDA, and on MLX on Metal. Products go to cuBLASLt with the bias and activation in its epilogue,
  and attention to cuDNN's fused kernel (or CUTLASS's batched FP16 GEMMs, or MLX); see
  [NATIVE.md](NATIVE.md).

```java
Tensor logits = Grad.inference(() -> {
    try (var half = Autocast.enableHalf()) {
        return model.forward(ids);                    // FP16 activations
    }
});
float[] out = logits.toFloatArray();                  // FP16 results read as floats
```

It needs gradients off; with gradients on, the scope behaves as `Autocast.enable(DType.F16)`.

## Kernels and memory traffic

`Engine.trafficStats()` returns running totals over every plan execution: `[executions, kernel
launches, library calls, bytes]`, the bytes being each operator's input and output tensors once
(the least it can read and write). The difference between two calls gives a step's counts, for
example to see what fusion (`-Djitnn.fuse`) saves:

```java
long[] before = Engine.trafficStats();
model.forward(x).toFloatArray();
long[] after = Engine.trafficStats();
System.out.printf("%d kernels, %.1f MB%n", after[1] - before[1], (after[3] - before[3]) / 1e6);
```

### Timeline

With `-Djitnn.profile=true` (and `-Djitnn.cudaGraph=false` on CUDA), `Engine.startTrace()` and
`Engine.stopTrace(file)` record a timeline as Chrome trace events. Open it in
[Perfetto](https://ui.perfetto.dev) or `chrome://tracing`.

```java
Engine.startTrace();
for (int step = 0; step < 10; step++) {
    try (AutoCloseable r = Engine.traceRange("step " + step)) {
        ...
    }
}
Engine.stopTrace(Path.of("trace.json"));
```

- "ranges": the spans opened with `Engine.traceRange`.
- "host: plan executions": every plan run, from the call to its return. The first run of a plan
  is marked, as it includes compiling the kernels.
- "device: kernels": every kernel, named after its operator, with its device time from
  TornadoVM's profiler. Kernels are drawn in launch order, back to back from the start of their
  plan's run, because TornadoVM reports durations, not start times. Their total equals the
  profiler's device time.

### Determinism

`Engine.setDeterministic(true)` (or `-Djitnn.deterministic=true`) makes runs reproducible bit for bit
on the same device, library versions and inputs, as PyTorch's `use_deterministic_algorithms`.

- jitNN's own kernels are deterministic already: they use no atomics, and split reductions add
  in a fixed order.
- The switch stops autotuning from measuring: choices follow the rules or the saved decisions,
  never this run's timings.
- It also asks cuDNN for deterministic algorithms only (with TornadoVM's cuDNN backward bindings).
  Set it before the first convolution, as cuDNN plans are cached.
- On an RTX 5070 Ti with vendor libraries and `-Djitnn.autotune=on`, two runs of three GPT-2
  training steps differed in 55% of the weights without the switch, and were identical with it.

## Autotuning

jitNN picks a kernel variant for each operator from rules measured on a few GPUs. With
`-Djitnn.autotune=on` it measures them on yours instead: the first time a shape runs, each
variant of its GEMMs, 1x1 and 3x3 convolutions and row norms is timed alone on random inputs,
checked against the rule's result, and the fastest is kept (only when more than 5% faster). The
decisions are saved per device in `~/.jitnn/autotune` and used by later runs, so the cost (a few
seconds per new shape, mostly compiling the variants) is paid once; `-Djitnn.autotune=use`
applies them without measuring new shapes. `Autotune.setMode` and `Autotune.setDirectory` do the
same from code.

## Serving

**Concurrent requests.** `-Djitnn.concurrency=N` lets up to N threads run plans on the device at
once.
- Each of N slots has its own compiled plans and device buffers, weights included, so memory
  grows with N.
- Threads in the same slot take turns.
- A plan that writes persistent tensors (parameters, optimiser state) runs while plans that do not
  touch them run too; plans that need those tensors wait for it.
- On an A100 with 4 threads and 4 slots, throughput is 2.1-2.7x at batch 1 and 1.1-1.35x at large
  batch.

**Release outputs once read.** An output stays valid after its plan runs again: before the next
run, jitNN copies it out of the plan's buffer into a host array of its own. Releasing the output
once it is read (`Tensor.release`, or try-with-resources) skips that copy, and the host arrays of
released or collected outputs are reused for later ones (up to `-Djitnn.hostPool.bytes`). On an
RTX 5070 Ti this takes BERT-base FP16 at batch 32 from 11.2 to 9.75 ms per call. Reading a released
tensor is an error.

```java
try (Tensor logits = Grad.inference(() -> model.forward(x))) {
    float[] out = logits.toFloatArray();
    // ...
}
```

**Request shapes.** jitNN compiles one plan per shape: kernels are specialised on their sizes, so
a new batch size or sequence length costs a compilation (seconds). `OnnxSession.withBuckets`
pads requests to a few shapes instead. On the A100, 40 encoder requests of random batch sizes and
lengths took 0.06 s with buckets, against 6-14 s compiling one plan per shape:

```java
OnnxSession session = OnnxSession.load(Path.of("encoder.onnx")).withBuckets(1, 2, 4, 8, 16, 32, 64, 128, 256, 512);
Map<String, Tensor> out = session.run(Map.of("input_ids", ids, "attention_mask", mask));   // any [batch, length]
```

- **What is padded:** the model's symbolic dimensions, to the smallest bucket that holds them. A
  size beyond the largest bucket runs as it is.
- **Only where padding is safe:**
  - a dimension that is the first axis of every input it appears in (the batch);
  - a dimension of an input whose name contains `mask` (a masked sequence).
- **Results:** inputs are padded with zeros and outputs cut back to the request's sizes, so they
  equal unpadded runs.
- **First use:** each bucket compiles on first use, so send one request per bucket at start-up.
- **Plan cache:** `withBuckets` reserves room in the plan cache for its buckets.
- **Any ONNX session:** `-Djitnn.onnx.buckets=1,2,4,...` gives every ONNX session buckets,
  DJL's and TensorFlow's included.

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

## Quantised weights from GGUF files

Weights in GGUF's block formats are used as they are stored: `Tensor.quantized` wraps the raw
blocks of a weight [out, in] (the bytes a GGUF file holds for that tensor), and `Gguf.linear`
multiplies by it without expanding it, with jitllm's kernels for one input row (decoding) or a
few (up to 32), and through a dequantised copy for larger batches.

```java
byte[] blocks = ...;                                             // a Q4_K tensor's bytes, 4096 x 4096
Tensor w = Tensor.quantized(DType.Q4_K, blocks, 4096, 4096);
Tensor y = Gguf.linear(x, w, null);                              // x [..., 4096] -> [..., 4096]
Tensor full = Gguf.dequantize(w);                                // FP32 copy, when needed
```

The gradient reaches `x` (through the dequantised weight), not the quantised weight. Q8_0 weights
go through the same call.

`jitnn-gguf` reads the weights from the file. `GgufFile.open` reads the metadata and the list of
tensors, and `tensor(name)` returns a persistent tensor: the block formats in their blocks, F32,
F16 and BF16 as those types, with shapes in jitNN's order (a GGUF weight listed as [in, out] is
[out, in], ready for `Gguf.linear`). Each tensor is memory-mapped copy-on-write rather than read,
so loading costs no copy and no Java heap and the file is never changed; a file that cannot be
opened for writing is copied instead (`tensor(name, true)` asks for a copy).

```java
GgufFile f = GgufFile.open(Path.of("Llama-3.2-1B-Instruct-Q4_K_M.gguf"));
int layers = f.integer("llama.block_count");
float eps = f.floatValue("llama.attention.layer_norm_rms_epsilon");
String[] tokens = f.strings("tokenizer.ggml.tokens");
Tensor wq = f.tensor("blk.0.attn_q.weight");                     // Q4_K [2048, 2048]
Tensor q = Gguf.linear(x, wq, null);
```

Tensors of types jitNN does not hold (Q8_1, Q8_K and the IQ formats) are listed, and
`raw(name)` gives their bytes, but they do not load as tensors.

### Text generation

The optional module `jitnn-llm` (built with `-Pllm`) runs a GGUF language model through jitllm,
which brings the tokenizers, the KV cache and the tuned kernels for Llama, Mistral, Qwen, Phi,
Granite, Gemma and other families:

```java
try (Llm llm = Llm.load(Path.of("Llama-3.2-1B-Instruct-Q8_0.gguf"))) {
    String answer = llm.generate("What is the capital of France?", 64);   // greedy
    llm.generate("Write a haiku about GPUs.", 64, System.out::print);      // streamed
    Tensor wq = llm.gguf().tensor("blk.0.attn_q.weight");                  // the same file's weights
}
```

`generate(GenerationRequest)` takes jitllm's requests (sampling, stop sequences, chat messages,
tools), and `model()` is the jitllm model. jitllm runs on TornadoVM's default device, which jitNN
also uses when `-Djitnn.device` is not set; TornadoVM's backend priorities choose it (for example
`-Dtornado.cuda.priority=100`).

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

The ranks' plans run at the same time, one per device. The default `HostCollective` moves tensors
through host memory, so the ranks can be on any mix of devices and backends (two GPUs, or CUDA and
OpenCL on the same GPU), but each collective waits for the host. On NVIDIA GPUs `NcclCollective`
keeps the collectives on the GPUs: they are recorded into each rank's plan like any operator (and
captured in its CUDA graph), and `averageGradients` packs every gradient into one buffer reduced
by one NCCL call, so a training step is one plan per rank:

```java
String[] devices = { "0:0", "0:1" };                            // distinct CUDA GPUs
try (NcclCollective nccl = new NcclCollective(devices)) {
    ProcessGroup.run(devices, nccl, (rank, collective) -> train(rank, collective));
    Engine.reset();                                              // release the plans before the communicator
}
```

With NCCL, results are new tensors (`allReduced`) or persistent tensors written in place
(parameters, buffers, `Tensor.persistent`).

`SyncBatchNorm2d(channels, collective, rank)` in place of `BatchNorm2d` normalises with the
statistics of the whole batch over the ranks (one differentiable NCCL all-reduce per layer), so
ranks training on shares of a batch match one device training on all of it.

Ranks may also be spread over processes, on one machine or several. One process makes an id and
hands it to the others (a file, an environment variable, the command line); each then joins with
its global ranks and devices, and `ProcessGroup.run` runs its own ranks:

```java
String id = NcclCollective.uniqueId();                          // in one process; pass it on
try (NcclCollective nccl = new NcclCollective(id, worldSize, firstRank, "0:0")) {
    ProcessGroup.run(new String[] { "0:0" }, nccl, (rank, collective) -> train(rank, collective));
    Engine.reset();
}
``` Tensors belong to the device they were computed on: do
not pass them between ranks except through the collective. `Engine.onDevice(spec, body)` runs a
block on another device from any thread.

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

## TensorFlow models

`jitnn-tf` runs TensorFlow SavedModels and frozen GraphDefs:

```java
TfModel model = TfModel.load(Path.of("exported_model"));              // a SavedModel directory, or a .pb file
Map<String, Tensor> out = model.run(Map.of("image", Tensor.of(pixels, 1, 224, 224, 3)));
```

Inputs and outputs carry the signature's names (`model.inputNames()`, `outputNames()`). Images
stay NHWC, as TensorFlow takes them; integer inputs are INT32 tensors.

## DJL

Add `jitnn-djl` and DJL code runs on jitNN unchanged:

```java
try (Model model = Model.newInstance("mlp")) {                     // jitNN is the default engine
    model.setBlock(new SequentialBlock().add(Linear.builder().setUnits(64).build()).add(Activation::relu)
            .add(Linear.builder().setUnits(10).build()));
    try (Trainer trainer = model.newTrainer(new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
            .optOptimizer(Optimizer.adam().build()))) {
        trainer.initialize(new Shape(32, 784));
        EasyTrain.fit(trainer, epochs, trainSet, validateSet);
    }
}
```

The engine is named `jitNN` and has rank 0, so it is DJL's default whenever `jitnn-djl` is on the
class path (`-Dai.djl.default_engine=PyTorch`, or naming the engine in `Model.newInstance`, picks
another). DJL's operators are recorded as jitNN traces, compiled at the first sync point (reading
the loss, for example) and replayed on later steps of the same shape. `Model.load` reads ONNX
files. NDArrays follow DJL's types: integer division truncates, and an integer array with a
floating scalar gives FLOAT32, as in PyTorch. Operators jitNN lacks run on the engine named by
`-Djitnn.djl.fallback` when it is on the class path. DJL's own examples (`TrainMnist`,
`TrainResnetWithCifar10`, `ImageClassification`, `YoloDetection`) run unchanged; see
[PERFORMANCE.md](PERFORMANCE.md#djl-examples).

## Custom operators

A custom operator is a kernel in plain Java, written with TornadoVM's `KernelContext`, plus an
optional host reference and gradient. Once built it behaves like a built-in operator: it is
recorded lazily, compiled into the same plan as its neighbours, cached and differentiated.

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

`CustomOp.unary` and `CustomOp.binary` cover element-wise FP32 operators: the kernel runs on a
1-D grid of at least `n` threads and bound-checks its index. `CustomOp.builder(name)` covers the
rest: the input types, one or more outputs with a shape function, the tasks to launch (each with an
explicit grid from `Emitter`), the host reference and the gradient.

```java
static final CustomOp MIN_MAX = CustomOp.builder("minMax")
        .inputs(DType.F32)
        .output((in, attrs) -> Shape.of(in[0].shape().dim(0)), DType.F32)
        .output((in, attrs) -> Shape.of(in[0].shape().dim(0)), DType.F32)
        .launch((e, n) -> {
            FloatArray x = e.f32(n.in(0));
            FloatArray min = e.f32(n.out(0));
            FloatArray max = e.f32(n.out(1));
            int rows = n.in(0).shape().dim(0);
            int cols = n.in(0).shape().dim(1);
            e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, MyKernels::minMax, e.context(), x, min, max, rows, cols));
        })
        .reference(n -> { /* the same on the host, into n.out(0) and n.out(1) */ })
        .build();

Tensor[] mm = MIN_MAX.applyAll(null, null, x);
```

`apply(iattr, fattr, inputs...)` passes integer and float attributes, which the launcher reads from
`n.iattr` and `n.fattr`. Compiled plans are cached by operator name and attributes, so names are
global: building a second operator with a name already in use throws. Without a reference the
operator runs only on a device; without a gradient, `backward()` through it throws. TornadoVM
names the generated kernel after the Java method, so a kernel method must not be named like a
CUDA or OpenCL built-in (`exp`, `hypot`, `max`, ...) or a C++ keyword (`not`, `and`, ...);
`unary` and `binary` reject such names.

## Printing the kernels

`Engine.printKernels(System.out)` prints every plan compiled from then on: its operators with
shapes and task numbers, then the CUDA, OpenCL or Metal source TornadoVM generates for each kernel,
custom ones included. `Engine.printKernels(null)` stops it; `-Djitnn.printKernels=true` (standard
output) or `-Djitnn.printKernels=<file>` turns it on at start-up. Plans already compiled are not
printed again: `Engine.reset()` drops them.

```
// ==== jitNN plan trace on CUDA NVIDIA A100 80GB PCIe: 2 operators, 2 kernels in 1 graph(s)
//   [0] custom:swish [77, 3] F32 -> [77, 3] F32  tasks t0
//   [1] binary [0] [77, 3] F32, [] F32 -> [77, 3] F32  tasks t1
extern "C" __global__ void swish(...)
```

## Vendor libraries

By default every operator runs on jitNN's own kernels. With `-Djitnn.native=auto` (the
`--with-native` option of jitNN's tools sets it), matrix products, convolutions, softmax,
attention in inference, FFTs and sparse products call cuBLAS, cuDNN, cuFFT and cuSPARSE on CUDA,
or MLX on Metal, as library tasks inside the same compiled plans:

```bash
java @$TORNADOVM_HOME/tornado-argfile -Djitnn.native=auto ...         # every library available
java @$TORNADOVM_HOME/tornado-argfile -Djitnn.native=cublas,cudnn ...  # only these
```

A library that is missing or fails is reported once, and its operators keep jitNN's kernels.
[NATIVE.md](NATIVE.md) lists what each library takes, the numerics (cuDNN convolutions use TF32 on
Ampere unless `NVIDIA_TF32_OVERRIDE=0`) and the requirements.

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
| `... -Djitnn.test.extra=-Djitnn.native=auto` | runs the device tests with the vendor libraries, and adds `NativeAccelTest` (each library-backed operator against the reference, and the fallback) |
| `scripts/bench/run_kernels.sh <device> <dir> [--with-native]`, `run_models.sh` | the kernel and model benchmarks of `jitnn-bench` |
