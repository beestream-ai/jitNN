# Operators

Every operator has a forward kernel, a gradient (a backward kernel or a composition of operators),
and a double-precision host reference. The order of implementation follows the operator plan:
waves ranked by how many of 11 reference models use each operator, with operators that need new
TornadoVM features last.

## Wave 1 (done)

| Operator | Kernels (`jitnn-core/.../kernels`) | Gradient |
|---|---|---|
| add, sub, mul, div, maximum, minimum, comparisons, with broadcasting | `ElementwiseKernels.binary`, `binaryBroadcast` | sum over broadcast axes |
| neg, exp, log, sqrt, abs, relu, pow, clamp, reciprocal, square, affine | `ElementwiseKernels.unary` | element-wise rules |
| sum, mean, max, min along an axis; full sums | `ReduceKernels.reduceGroup`, `reduceThread`, `reduceRowsSplit`, `reduceColumnsSplit` | broadcast back; extremum mask |
| argmax | `ReduceKernels.argmaxThread` | none |
| reshape | a view, no kernel | reshape back |
| permute, transpose | `ElementwiseKernels.permute`, `transposeTiled32` | inverse permutation |
| matmul with either transpose | `GemmKernels.gemm`, `gemm64`, `gemm128`, `gemmSimdgroup` | two GEMMs |
| linear (+ bias, + ReLU, fused) | the GEMMs with an epilogue, `biasActivation` on Metal | GEMMs, ReLU mask, bias sum |
| softmax, log-softmax (last axis) | `SoftmaxKernels.softmaxRowsCached`, `softmaxRows` | `softmaxBackwardRows` |
| cross-entropy (fused, mean) | `crossEntropyRows` + mean | `crossEntropyBackwardRows` |
| NLL | `nllRows` | `nllBackward` |
| uniform, normal (Philox4x32-10) | `RandomKernels.philox` | none |
| SGD, SGD momentum and Nesterov, Adam, AdamW | `OptimKernels` | none |

## Next waves

| Wave | Contents | Gate |
|---|---|---|
| 2 | Conv2D (with groups and depthwise), BatchNorm, pooling, dropout, LR schedules | LeNet, ResNet-50, MobileNetV2 |
| 3 | embedding, LayerNorm, RMSNorm, GELU, SiLU, attention with masks, RoPE, AdamW schedules, FP16/BF16 | BERT, GPT-2, a Llama-style decoder |
| 4 | sigmoid, LSTM, GRU, transposed conv, upsampling, BCE, MSE, KL | LSTM LM, U-Net, DCGAN, VAE |
| 5 | the rest of the DL4J/DJL activations, losses, optimisers and norms, Conv1D/3D, sort, scatter | the DL4J and DJL example suites |
| 6 | decompositions, sparse and graph ops, FFT, Word2Vec, classical ML | 1.0 |
| 7 | operators blocked on TornadoVM features (data-dependent shapes, FP8 off CUDA, multi-GPU) | as each feature lands |
