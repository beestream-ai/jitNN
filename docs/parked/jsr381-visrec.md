# Parked: jitNN as a JSR 381 (javax.visrec) implementation

`jitnn-visrec` makes jitNN a provider for JSR 381, the Java visual recognition API
(`javax.visrec:visrec-api` 1.0.5). Parked: the API is dormant and its only other implementation is
Deep Netts' reference implementation, and jitNN's own API, the ONNX and TensorFlow importers and the
DJL engine already cover image classification better. Pick it up if someone needs JSR 381.

## What works

- `JitnnServiceProvider`: found by `ServiceProvider.current()` (priority 100, ahead of the reference
  implementation's 0); `BufferedImage` images through ImageIO.
- `NeuralNetImageClassifier.builder()` (`JitnnImageClassifierFactory`): reads what the reference
  implementation reads, a network in Deep Netts' JSON format (INPUT, CONVOLUTIONAL, MAXPOOLING,
  FULLY_CONNECTED, OUTPUT with SIGMOID or SOFTMAX), a labels file and an image index of
  `<path> <label>` lines; trains with SGD (momentum 0.7); `exportModel` writes one safetensors file
  with the weights, the architecture and the labels, and `importModel` reads it.
- `NeuralNetBinaryClassifier.builder()` (`JitnnBinaryClassifierFactory`): an MLP with sigmoid hidden
  layers trained from a CSV file.
- `VisRecTest` goes through the JSR 381 API only: provider lookup, binary classifier (192/200), a
  CNN trained, exported and imported (same scores).

## Measured (A100 host, MNIST PNGs: 6,000 train, 1,000 test, 3 epochs; CSV: 10,000 rows, 20 features)

| | image training | classify per image | image accuracy | binary classify per row | binary accuracy |
|---|---|---|---|---|---|
| Deep Netts reference | 114.2 s | 1.31 ms | 91.0% | 2.2 us | 96.0% |
| jitNN, CPU backend | 7.6 s | 0.34 ms | 85.4% | 19.9 us | 99.0% |
| jitNN, CUDA | 8.5 s | 0.24 ms | 85.4% | 47.0 us | 99.0% |

## Open

- Accuracy: jitNN trains in batches of 32 where the reference updates after every example (564
  against 18,000 updates in 3 epochs); a batch-size setting (default 1) or a learning rate scaled by
  the batch size would match it.
- One input per `classify` call: jitNN's plan overhead (20-47 us) loses to Deep Netts' plain Java on
  tiny networks; small models could take the reference path, or a batched classify could be added.
- Not implemented: `ObjectDetector`, `LogisticRegression`, `SimpleLinearRegression`, evaluators.
- In the JSR 381 API itself: `ServiceProvider.setCurrent` throws (`Collections.copy` into an empty
  list), and with the reference implementation also present, the factory used depends on class-path
  order.
