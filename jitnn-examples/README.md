# jitNN examples

Small, complete programs that show what jitNN feels like to use: well-known examples from DL4J,
Deep Netts, DJL, PyTorch, Keras, H2O, Tribuo and Smile, written again with jitNN's API (one is DJL
code that runs on jitNN unchanged). Each downloads its data on first use and prints what it learned.

## Running them

```bash
source <TornadoVM>/setvars.sh            # TORNADOVM_HOME, as for building jitNN
./mvnw install -DskipTests               # in the jitNN checkout, once
jitnn-examples/run.sh                    # lists the examples
jitnn-examples/run.sh MnistMlp           # runs one; arguments follow the name
jitnn-examples/run.sh Gpt2TextGeneration "The meaning of life is" 80
```

- Downloads go to `~/.jitnn/examples` (`JAVA_OPTS=-Djitnn.examples.data=<dir>` to change it).
- `JAVA_OPTS` passes JVM flags: `-Djitnn.device=1:0` for another device (`tornado --devices` lists
  them), `-Djitnn.verbose=true` to log every compiled plan, `-Djitnn.withNative=true` to let jitNN
  call cuBLAS and cuDNN.
- `JITNN_DEVICE_MEMORY` sets TornadoVM's device-memory budget (default 12GB).

The first step of every example compiles the whole step (forward, backward and optimiser) into one
plan, so the first epoch takes longer; later steps replay the plan.

## The examples

Results are with the default arguments.

### Neural networks

| Example | After | What it does | Result |
|---|---|---|---|
| `MnistMlp` | DL4J MLPMnistTwoLayerExample, Deep Netts' MNIST | a 784-500-100-10 perceptron on handwritten digits, Adam | 97.4% test accuracy in 3 epochs |
| `MnistCnn` | PyTorch's MNIST example | two convolutions, max pooling and dropout, AdaDelta (lr 1.0) with the rate times 0.7 every epoch, 14 epochs | 99.2% test accuracy |
| `LeNetMnist` | DL4J LeNetMNIST | two convolutions with max pooling, SGD with Nesterov momentum | 99.0% in 3 epochs |
| `IrisClassifier` | DL4J IrisClassifier, Deep Netts' iris | a 4-16-3 network against a random forest and logistic regression from `ai.beestream.jitnn.ml` | 30 of 30 test flowers for all three |
| `Cifar10ResNet` | DJL TrainResnetWithCifar10 | ResNet-18 with random crops and flips, batches assembled by four threads | 66.3% after 2 epochs |
| `CharLstmShakespeare` | DL4J LSTMCharModellingExample | a two-layer LSTM over the bytes of Tiny Shakespeare, then samples text one character at a time with its state | Shakespeare-shaped text after 2,000 steps |
| `NanoGptShakespeare` | Andrej Karpathy's nanoGPT (shakespeare_char) | nanoGPT's settings: a 6-layer, 6-head, 384-wide GPT with 256 tokens of context, AdamW with warm-up and cosine decay over 5,000 steps, then a new scene | validation loss 1.69 after 5,000 steps (lowest 1.47 at step 2,000) |
| `VaeMnist` | DL4J VariationalAutoEncoderExample, PyTorch's VAE | PyTorch's settings: a 784-400-20 variational autoencoder, 10 epochs; decodes 64 random latent points into `vae-digits.png` | test loss 105.5 per image |
| `DcganMnist` | PyTorch's DCGAN (`--dataset mnist`) | MNIST at 64x64, five transposed convolutions against five convolutions, Adam for both, 25 epochs | generated digits in `dcgan-digits.png` |
| `Word2VecShakespeare` | DL4J Word2VecRawTextExample | skip-gram with negative sampling; nearest neighbours of a few words | "father": mother, guest, ... |
| `GcnCora` | PyTorch's GCN (Kipf and Welling) | the original's settings: two graph convolutions on the Cora citation graph, 1,000 test papers, 500 for validation, 200 epochs | 77.6% test accuracy |
| `KerasImdbSentiment` | Keras' text classification from scratch | an embedding, strided 1-D convolutions and max pooling on 50,000 film reviews | 84.5% test accuracy in 3 epochs |
| `KerasMovieLens` | Keras' collaborative filtering | user and film embeddings on 100,836 ratings; recommends films for a user | validation error 0.89 stars after 5 epochs |

### Pretrained models

| Example | After | What it does | Result |
|---|---|---|---|
| `OnnxImageClassification` | DJL ImageClassification | ResNet-50 from the ONNX model zoo, run on jitNN's kernels | "tabby" 62.5% on DJL's kitten |
| `BertFillMask` | BERT's masked-word task | bert-base-uncased from Hugging Face fills in `[MASK]` | "The capital of France is [MASK]." gives paris |
| `Gpt2TextGeneration` | DJL TextGeneration | GPT-2 (124M) from Hugging Face continues a prompt; one compiled plan serves every token | the continued text |

### Classical machine learning

| Example | After | What it does | Result |
|---|---|---|---|
| `H2oProstate` | H2O's GLM, GBM and random-forest walkthroughs | predicts capsule penetration of 380 prostate tumours | AUC: logistic regression 0.845, gradient boosting 0.837 |
| `TribuoWineQuality` | Tribuo's regression tutorial | ridge regression, gradient-boosted trees and an early-stopped network on 1,599 red wines | R²: gradient boosting 0.376, ridge 0.316, network 0.272 |
| `SmileTsneMnist` | Smile's t-SNE | t-SNE of 2,000 digits on the device, drawn into `tsne-mnist.png` | 83% of each point's nearest neighbours share its digit |

### jitNN features

| Example | What it does | Result |
|---|---|---|
| `CustomOpSwish` | a swish activation written as a TornadoVM `KernelContext` kernel, with a gradient; checked against the built-in `silu`, then trains an MNIST network | matches `silu` to 5e-7; 97.7% |
| `DataParallelCifar10` | ResNet-18 on several devices at once (`--nccl` on NVIDIA GPUs), 128 images per device, then on one device | test accuracy and the scaling over one device |
| `djl.TrainMnistDjl` | DJL's TrainMnist: DJL's `Mlp`, dataset and `EasyTrain`, with jitNN as DJL's engine | 96.8% in 2 epochs |

Timings are in [jitNN-benchmarking](https://github.com/beestream-ai/jitNN-benchmarking).

## Data

The examples download, on first use: MNIST, CIFAR-10, Tiny Shakespeare (from char-rnn), Cora
(LINQS), the IMDB reviews (fast.ai's copy of Maas et al.), MovieLens ml-latest-small (GroupLens),
the prostate data (H2O), the red wine quality data (UCI), ResNet-50 from the ONNX model zoo, and
GPT-2 and bert-base-uncased from Hugging Face. The iris data (UCI, CC BY 4.0, see
[NOTICE](../NOTICE)) ships with the module.
