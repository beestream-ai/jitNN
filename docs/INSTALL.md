# Installing and building jitNN

## Requirements

| What | Version | Notes |
|---|---|---|
| JDK | 22 or newer | jitNN compiles with `--release 22`; tested on Temurin 25.0.3 and JDK 25.0.2 |
| TornadoVM | `develop`, 7.2.1-jdk22plus-dev or newer | needs the on-demand `TornadoExecutionPlan.transferToDevice(Object...)` and the library tasks (`TaskGraph.libraryTask`) with their modules (`tornado-cublas`, `tornado-cudnn`, `tornado-cufft`, `tornado-cusparse`, `tornado-mlx`), which every `develop` build installs; earlier versions lack the library tasks |
| A device | CUDA, OpenCL or Metal | optional: the reference backend runs on any CPU |
| cuDNN, MLX | cuDNN 9; MLX from `brew install mlx` | optional: only for `-Djitnn.native` (see [NATIVE.md](NATIVE.md)); cuBLAS, cuFFT and cuSPARSE come with the CUDA toolkit |
| Python, `torch`, `numpy` | any recent | optional: only to regenerate the PyTorch references |
| `gguf` (llama.cpp's Python package) | any recent | optional: only to regenerate the goldens for GGUF's quantised formats |
| `scikit-learn` | 1.9.1 for the committed references | optional: only to regenerate the classical machine-learning references |
| `onnx`, `onnxruntime` | any recent | optional: only to regenerate the ONNX goldens and for the inference comparison |
| `tensorflow` 2.x with Keras 3 | any recent | optional: only to regenerate the TensorFlow goldens |

jitNN itself has no dependencies besides TornadoVM's `tornado-api` and its library modules, all
provided by the TornadoVM SDK at run time. `jitnn-onnx` reads ONNX files with its own protobuf reader,
which `jitnn-tf` also uses for TensorFlow models. `jitnn-djl` depends on DJL's `ai.djl:api` (0.38.0).

## TornadoVM

Build TornadoVM from `develop` with the backends you want and install its artifacts into the local
Maven repository, which jitNN resolves `tornado-api` and the library modules from:

```bash
git clone -b develop https://github.com/beehive-lab/TornadoVM && cd TornadoVM
make jdk22plus BACKEND=opencl,metal      # or BACKEND=cuda, or several
source setvars.sh                        # JAVA_HOME, TORNADOVM_HOME, PATH
tornado --devices                        # device ids, as <driver>:<device>
```

`source setvars.sh` is needed in every shell that builds or runs jitNN.

## jitNN

```bash
git clone https://github.com/beehive-lab/jitnn && cd jitnn
./mvnw install -DskipTests
./mvnw test                                          # reference backend only
./mvnw test -Paccel-tests -Djitnn.test.device=0:0    # plus the device tests
```

The Maven wrapper downloads Maven 3.9.6 on first use. Offline builds work with `./mvnw -o` once the
dependencies are in `~/.m2`.

## Test data

The MNIST gate needs the four MNIST IDX files (gzipped is fine) and, for the PyTorch comparison, a
reference run in the same directory:

```bash
mkdir -p ~/jitnn-data/mnist && cd ~/jitnn-data/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fLO https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
done
python <jitnn>/scripts/pytorch/mnist_mlp.py ~/jitnn-data/mnist ~/jitnn-data/mnist/mnist_mlp_pytorch.f32
./mvnw test -Paccel-tests -Djitnn.mnist=$HOME/jitnn-data/mnist
```

The CIFAR-10 gates need the binary version of CIFAR-10 (170 MB) and the PyTorch references, which
take a while on a CPU (a GPU is much faster: set `JITNN_TORCH_DEVICE=cuda` or `mps`):

```bash
mkdir -p ~/jitnn-data/cifar10 && cd ~/jitnn-data/cifar10
curl -fLO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz && tar xzf cifar-10-binary.tar.gz
for m in resnet18 mobilenetv2; do
  python <jitnn>/scripts/pytorch/cnn_train.py $m . ${m}_cifar10_pytorch.f32 0 cifar10
  python <jitnn>/scripts/pytorch/cnn_grad64.py $m . ${m}_cifar10_pytorch.f32 cifar10
done
./mvnw test -Paccel-tests -Djitnn.cifar10=$HOME/jitnn-data/cifar10 -Dtest='CnnAccelTest#*Cifar10*'
```

The transformer gates train on a byte-level corpus; any text works, and the results in the docs
use TornadoVM's documentation (`docs/source/*.rst` concatenated, 307 KB):

```bash
mkdir -p ~/jitnn-data/text && cat <TornadoVM>/docs/source/*.rst > ~/jitnn-data/text/corpus.txt
for m in gpt2 llama bert; do
  python <jitnn>/scripts/pytorch/transformers.py train  $m ~/jitnn-data/text/corpus.txt ~/jitnn-data/text/${m}_pytorch.f32 300
  python <jitnn>/scripts/pytorch/transformers.py grad64 $m ~/jitnn-data/text/corpus.txt ~/jitnn-data/text/${m}_pytorch.f32
done
./mvnw test -Paccel-tests -Djitnn.text=$HOME/jitnn-data/text -Dtest=TransformerAccelTest
```

## PyTorch environment

```bash
python3 -m venv ~/jitnn-data/torch-venv
~/jitnn-data/torch-venv/bin/pip install torch numpy
~/jitnn-data/torch-venv/bin/python scripts/pytorch/goldens.py jitnn-core/src/test/resources/golden
```

The golden set is committed, so this is only needed to change or extend it. The other references
are committed too and are regenerated the same way:

```bash
~/jitnn-data/torch-venv/bin/pip install scikit-learn onnx onnxruntime
~/jitnn-data/torch-venv/bin/python scripts/sklearn/ml_goldens.py jitnn-nn/src/test/resources/ml
~/jitnn-data/torch-venv/bin/python scripts/onnx/onnx_goldens.py jitnn-onnx/src/test/resources/onnx
```

The TensorFlow goldens need TensorFlow (CPU is enough), best in a separate environment:

```bash
python3 -m venv ~/jitnn-data/tf-venv
~/jitnn-data/tf-venv/bin/pip install tensorflow numpy
~/jitnn-data/tf-venv/bin/python scripts/tf/tf_goldens.py jitnn-tf/src/test/resources/tf
```
