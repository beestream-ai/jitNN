# Validation

Five layers of checks run on the host and on every device.

1. **Host reference.** Every operator has a double-precision Java implementation, written
   independently of its kernel. `Engine.Backend.REFERENCE` runs it; it is the oracle the device
   tests compare against.
2. **Finite differences.** `GradientTest` compares every gradient with central differences:
   broadcasting binaries, every unary, reductions, all four GEMM transposes, permute, reshape,
   softmax, log-softmax, cross-entropy, fused linear and an MLP chain. It runs in FP32 with a loose
   tolerance, as jitNN has no FP64 type yet.
3. **PyTorch goldens.** `scripts/pytorch/goldens.py` computes 449 cases in FP64 (inputs, output and
   every input gradient, on odd, size-1 and broadcast shapes):
   - 435 operator cases: element-wise, reductions, GEMM, convolution, BatchNorm, pooling,
     transformer activations, slicing and joining, embedding, LayerNorm and RMSNorm, batched
     matmul, attention with causal and padding masks, multi-head attention against
     `nn.MultiheadAttention`, RoPE, top-k, FP16/BF16 casts and half-precision GEMMs, the losses,
     transposed convolution, resizing, LSTM and GRU against PyTorch's, the extended operator set,
     CTC and metric losses, linear algebra against `torch.linalg`, Fourier transforms against
     `torch.fft`, sparse products, nonzero, masked select, unique, FP8, and linear layers and
     dequantisation on GGUF's Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K and Q3_K blocks
     (random blocks dequantised by llama.cpp's `gguf` package);
   - 14 optimiser cases: three steps of each `torch.optim` optimiser, and of PyTorch
     implementations in the script for LAMB and LARS;
   - Philox values reproduced in numpy.

   `GoldenTest` checks the reference against them; `GoldenAccelTest` and `OptimGoldenAccelTest`
   check every device.
4. **Other references.**

   | Test | Checks |
   |---|---|
   | `MlAccelTest` | classical machine-learning estimators against scikit-learn on stored datasets (`scripts/sklearn/ml_goldens.py`) |
   | `OnnxGoldenTest`, `OnnxGoldenAccelTest` | ONNX models against ONNX Runtime's outputs (`scripts/onnx/onnx_goldens.py`) |
   | `TfGoldenTest`, `TfGoldenAccelTest` | TensorFlow SavedModels (MLP, CNN, transformer block, operator mix) and a frozen GraphDef against TensorFlow (`scripts/tf/tf_goldens.py`) |
   | `DjlEngineTest`, `DjlEngineAccelTest` | DJL programs as DJL users write them (MLPs with Adam and SGD, a CNN with BatchNorm, an LSTM, NDArray semantics, ONNX through `Model.load`) on the reference, then step by step against it on a device |
   | `NativeAccelTest` | with `-Djitnn.native`: every operator that calls a vendor library, forward and backward, and the fallback on shapes the libraries cannot take |
   | `HalfInferenceAccelTest` | FP16 inference (`Autocast.enableHalf`: encoder blocks, attention) against the FP32 host reference, with or without vendor libraries |
   | `FusionAccelTest` | fused inference (convolution + BatchNorm, residual add + ReLU, linear + GELU) against the unfused host reference |
   | `AutotuneAccelTest` | every measured variant reproduces the host reference; saved decisions are read back |
   | `GgufFileTest`, `GgufAccelTest`, `LlmAccelTest` | a fixture written by llama.cpp's `gguf` package (`scripts/gguf/gguf_fixture.py`), mapped and copied; `LlmAccelTest` (`-Pllm`) generates text from a GGUF model through jitllm |
   | `DataParallelAccelTest`, `DataParallelMultiProcessAccelTest` | ranks in one process on host collectives and NCCL (`-Djitnn.dist.nccl=true`), and two processes over NCCL, against single-device training on the full batch |

5. **Model parity.** Models start from bit-identical weights (`scripts/pytorch/jitnn_init.py`
   reimplements jitNN's Philox initialisation; `models.py` builds ResNet-18 and MobileNetV2 with the
   same seeds) and train on the same batches as PyTorch.
   - **MNIST gates** (MLP, 2 epochs; LeNet-5, 1 epoch): losses of steps 0-9 within 1e-4, steps
     0-49 within 1%, the mean of the last 100 within 5%, accuracy within 0.3 points.
   - **CNN gates** (ResNet-18 and MobileNetV2: 200 steps on MNIST, and one epoch of 781 steps on
     CIFAR-10 evaluated on all 10,000 test images). Loss curves of deep BatchNorm networks drift
     apart within a few Adam steps from FP32 rounding alone, so the gate compares every parameter's
     step-0 gradient with PyTorch in FP64 (`cnn_grad64.py`). A parameter passes within three times
     PyTorch's own FP32 error on it, relative to its gradient or, for gradients that nearly cancel,
     to the model's median absolute FP32 error. Then the step-0 loss (1e-5), the mean loss of the
     last 50 steps (10%) and the evaluation accuracy (3 points).
   - **Transformer gates** (`TransformerAccelTest`, 300 steps on a byte-level corpus): GPT-2 and a
     Llama-style decoder as language models, BERT with masked language modelling and padding
     masks; 4 layers of 128 features, 4 heads, sequences of 128, batch 16; AdamW with warm-up,
     cosine decay and gradient clipping at 1. Step-0 gradients as for the CNN gates, the step-0
     loss (1e-5), the mean of the last 50 losses (5%) and the evaluation loss (5%).
   - **Mixed-precision gates** (GPT-2 under BF16 autocast, Llama and BERT under FP16 autocast with
     dynamic loss scaling) against the FP32 PyTorch run: half-precision operands cannot follow FP32
     step by step, so the step-0 loss is checked to 1e-2 and the end of training to 3%.
   - **Generative and recurrent gates** (`scripts/pytorch/generative_recurrent.py`; same initial
     weights, batches and Philox noise as PyTorch): a VAE on MNIST (BCE plus Gaussian KL, 400
     steps), a DCGAN on MNIST (transposed convolutions, LeakyReLU, BCE with logits, two Adam
     optimisers, 300 steps) and a U-Net segmenting MNIST digits (300 steps) in
     `GenerativeAccelTest`; two-layer LSTM and GRU byte-level language models against cuDNN's (300
     steps) in `RecurrentAccelTest`. Step-0 gradients as above, the step-0 loss (1e-5), the mean of
     the last 50 losses (2% VAE, 3% language models, 10% DCGAN and U-Net, whose loss ends near 0.02
     where FP32 summation order alone moves the curve by 5-7%) and the evaluation metric.

Tolerances: element-wise operators 2e-6 relative; reductions, GEMM, softmax and losses 1e-4;
gradients ten times those.

## Results

### Core, CNN and transformer gates

Golden counts cover the core, CNN and transformer operators. "Gradient" is the worst step-0
gradient as a fraction of its bound; "last-50 loss" the mean loss of the last 50 steps. Without
BatchNorm amplifying rounding, the transformer runs stay within 1e-7 of PyTorch for their first
steps and end on the same losses to four digits.

| Check | Reference | OpenCL (M4 Pro) | Metal (M4 Pro) | CUDA (A10) |
|---|---|---|---|---|
| Operator goldens (171) | pass | pass | pass | pass |
| Optimiser goldens (5) | pass | pass | pass | pass |
| Philox against numpy | bit-exact | bit-exact | bit-exact | bit-exact |
| MLP gate: loss vs reference at every step, 1e-3 | | pass | pass | pass |
| MNIST: worst relative loss gap, steps 0-49 | | 5.0e-7 | 7.0e-7 | 5.0e-7 |
| MNIST MLP: test accuracy (PyTorch: 96.24%) | | 96.15% | 96.14% | 96.25% |
| LeNet-5: test accuracy (PyTorch: 96.87%) | | 96.93% | 97.02% | 97.00% |
| ResNet-18 (MNIST): gradient / accuracy after 200 steps (PyTorch: 93.2%) | | 0.38 / 93.0% | 0.33 / 92.7% | 0.85 / 93.9% |
| MobileNetV2 (MNIST): gradient / accuracy after 200 steps (PyTorch: 92.3%) | | 0.64 / 94.0% | 0.60 / 91.5% | 0.69 / 94.9% |
| CIFAR-10 ResNet-18: gradient / last-50 loss / accuracy (PyTorch: 0.946 / 66.0%) | | 0.23 / 0.951 / 67.5% | 0.26 / 0.922 / 66.5% | 0.14 / 0.945 / 63.9% |
| CIFAR-10 MobileNetV2: gradient / last-50 loss / accuracy (PyTorch: 1.118 / 59.2%) | | 0.19 / 1.120 / 57.8% | 0.22 / 1.144 / 60.9% | 0.24 / 1.137 / 58.3% |
| GPT-2: gradient / last-50 loss / evaluation loss (PyTorch: 3.0009 / 3.1581) | | 0.06 / 3.0009 / 3.1581 | 0.06 / 3.0009 / 3.1581 | 0.06 / 3.0009 / 3.1581 |
| Llama: gradient / last-50 loss / evaluation loss (PyTorch: 1.9951 / 2.5854) | | 0.07 / 1.9951 / 2.5854 | 0.07 / 1.9951 / 2.5854 | 0.07 / 1.9951 / 2.5854 |
| BERT: gradient / last-50 loss / evaluation loss (PyTorch: 3.7603 / 3.4829) | | 0.06 / 3.7603 / 3.4831 | 0.06 / 3.7603 / 3.4831 | 0.06 / 3.7603 / 3.4831 |
| GPT-2 BF16: last-50 loss / evaluation loss (FP32 PyTorch: 3.0009 / 3.1581) | | 3.0014 / 3.1565 | 3.0016 / 3.1561 | 3.0017 / 3.1558 |
| Llama FP16 + loss scaling: same (FP32 PyTorch: 1.9951 / 2.5854) | | 1.9951 / 2.5854 | 1.9952 / 2.5858 | 1.9951 / 2.5853 |
| BERT FP16 + loss scaling: same (FP32 PyTorch: 3.7603 / 3.4829) | | 3.7606 / 3.4842 | 3.7617 / 3.4851 | 3.7608 / 3.4818 |

### Generative and recurrent gates

Against PyTorch references made on the same GPU; the A10 passes the same gates against references
made on it. Metal has not run these yet.

| Check | PyTorch | CUDA (A100) | OpenCL (A100) |
|---|---|---|---|
| VAE: gradient | | 0.04 | 0.04 |
| VAE: last-50 loss / test loss | 140.89 / 134.87 | 140.87 / 134.78 | 140.91 / 134.81 |
| DCGAN: gradient, discriminator / generator | | 0.17 / 0.08 | 0.17 / 0.06 |
| DCGAN: last-50 loss, discriminator / generator | 0.325 / 2.398 | 0.318 / 2.372 | 0.319 / 2.430 |
| U-Net: gradient | | 0.02 | 0.02 |
| U-Net: test loss / foreground IoU | 0.0207 / 0.933 | 0.0214 / 0.932 | 0.0215 / 0.931 |
| LSTM LM: gradient | | 0.04 | 0.05 |
| LSTM LM: last-50 loss / evaluation loss | 3.7204 / 3.6513 | 3.7204 / 3.6513 | 3.7204 / 3.6513 |
| GRU LM: gradient | | 0.03 | 0.03 |
| GRU LM: last-50 loss / evaluation loss | 3.4299 / 3.4294 | 3.4190 / 3.4366 | 3.4022 / 3.4480 |

Every gate of the previous table also passes on both A100 backends.

### Full suite

On both A100 backends. Metal has not run the linear algebra, Fourier, sparse, embedding, classical
machine-learning, FP8, collective and ONNX checks yet.

| Check | Reference | CUDA (A100) | OpenCL (A100) |
|---|---|---|---|
| PyTorch golden set: 435 operator cases | pass | pass | pass |
| PyTorch optimiser goldens (14) | pass | pass | pass |
| ONNX goldens (78 models) against ONNX Runtime (CPU) | pass | pass | pass |
| Model gates (56; two inference benchmarks skipped by default) | | pass | pass |
| scikit-learn comparisons (`MlAccelTest`) | | pass | pass |
| Embedding gates | | pass | pass |
| Data-parallel gate | | pass | pass |

The ONNX goldens are 76 single-operator models built with the `onnx` package, plus a CNN and a
transformer block exported from PyTorch.

#### Classical machine learning against scikit-learn 1.9.1

`MlAccelTest` fits each estimator on the stored data and compares with scikit-learn's result.

| Estimator | Agreement with scikit-learn |
|---|---|
| k-means from given centres | centres within 1.4e-6, identical labels |
| k-means++ | inertia 4261.40 (scikit-learn 4261.40) |
| kNN | identical classes and probabilities; regression within 2e-6 |
| PCA | components within 1.2e-6 |
| Least squares, ridge | coefficients within 1e-6 |
| Logistic regression | coefficients within 6.5e-4, probabilities within 1.1e-4 |
| Linear SVM | coefficients within 3e-6 |
| Naive Bayes | probabilities within 3e-6 |

Trees and t-SNE depend on random choices that differ between the libraries, so they are compared
by held-out quality:

| Model | jitNN | scikit-learn |
|---|---|---|
| Decision tree, accuracy | 0.815 | 0.800 |
| Random forest, accuracy | 0.875 | 0.875 |
| Gradient boosting, accuracy | 0.870 | 0.860 |
| Decision tree, R^2 | 0.690 | 0.636 |
| Random forest, R^2 | 0.849 | 0.839 |
| Gradient boosting, R^2 | 0.938 | 0.931 |
| t-SNE, trustworthiness | 0.960 (CUDA), 0.965 (OpenCL) | 0.967 |

#### Embedding models

`EmbeddingModelsAccelTest` trains each model on a corpus with four planted topics and checks that
the loss falls and that each word's (or document's) nearest neighbour shares its topic:

| Model | Loss, first to last steps | Nearest-neighbour purity |
|---|---|---|
| word2vec skip-gram | 4.14 -> 1.58 | 1.000 |
| word2vec CBOW | 4.13 -> 2.32 | 1.000 |
| GloVe | 3.43 -> 0.0005 | 1.000 |
| doc2vec PV-DBOW | 4.16 -> 1.52 | 1.000 |
| doc2vec PV-DM | 4.14 -> 1.57 | 1.000 |

#### Data-parallel training

`DataParallelAccelTest` trains two ranks on half batches and one rank on full batches from the
same initial weights. Losses and final parameters agree within 2e-5 with both ranks on CUDA, both
on OpenCL, and one on each on the same A100.
