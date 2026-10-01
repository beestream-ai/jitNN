# Installing and building jitNN

## Requirements

| What | Version | Notes |
|---|---|---|
| JDK | 22 or newer | jitNN compiles with `--release 22`; tested on Temurin 25.0.3 and JDK 25.0.2 |
| TornadoVM | `develop`, 7.0.2-jdk22plus-dev | needs the on-demand `TornadoExecutionPlan.transferToDevice(Object...)`, which is not in 7.0.1 |
| A device | CUDA, OpenCL or Metal | optional: the reference backend runs on any CPU |
| Python, `torch`, `numpy` | any recent | optional: only to regenerate the PyTorch references |

## TornadoVM

Build TornadoVM from `develop` with the backends you want and install its artifacts into the local
Maven repository, which jitNN resolves `tornado-api` from:

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

## PyTorch environment

```bash
python3 -m venv ~/jitnn-data/torch-venv
~/jitnn-data/torch-venv/bin/pip install torch numpy
~/jitnn-data/torch-venv/bin/python scripts/pytorch/goldens.py jitnn-core/src/test/resources/golden
```

The golden set is committed, so this is only needed to change or extend it.
