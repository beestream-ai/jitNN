# Installing and building jitNN

## Requirements

| What | Version | Notes |
|---|---|---|
| JDK | 22 or newer | compiled with `--release 22`; tested on JDK 25 |
| TornadoVM | 8.1.0 or newer | needs `TornadoExecutionPlan.transferToDevice(Object...)`, `withDynamicShapes()` and the library tasks (`TaskGraph.libraryTask`) with their modules (`tornado-cublas`, `tornado-cudnn`, `tornado-cufft`, `tornado-cusparse`, `tornado-mlx`), all on Maven Central |
| A device | CUDA, OpenCL or Metal | optional: the reference backend runs on any CPU |
| CUDA toolkit | 12 or 13; 13 for Blackwell GPUs (RTX 50 series) | optional, for the CUDA backend |
| cuDNN, MLX | cuDNN 9; MLX from `brew install mlx` | optional, for `-Djitnn.native` (see [NATIVE.md](NATIVE.md)); cuBLAS, cuFFT and cuSPARSE come with the CUDA toolkit |
| Python packages | `torch`, `numpy`, `gguf` (llama.cpp's), `onnx`, `onnxruntime`: any recent; `scikit-learn` 1.9.1; `tensorflow` 2.x with Keras 3 | optional, only to regenerate the [references](#regenerating-the-references) |

jitNN depends only on TornadoVM's `tornado-api` and library modules, which Maven fetches from
Maven Central; the TornadoVM SDK provides them at run time. `jitnn-onnx` and `jitnn-tf` use their own protobuf reader, `jitnn-gguf` its own GGUF
reader. `jitnn-djl` depends on DJL's `ai.djl:api` (0.38.0). The optional `jitnn-llm`
(`./mvnw -Pllm ...`) needs jitllm built against the same TornadoVM SDK: run `./mvnw install` in a
jitllm checkout first.

## TornadoVM

Build the TornadoVM SDK, the runtime jitNN runs on, with the backends you want; `v8.1.0` matches
the version jitNN compiles against:

```bash
git clone -b v8.1.0 https://github.com/beehive-lab/TornadoVM && cd TornadoVM
make jdk22plus BACKEND=opencl,metal      # or BACKEND=cuda, or several
source setvars.sh                        # JAVA_HOME, TORNADOVM_HOME, PATH
tornado --devices                        # device ids, as <driver>:<device>
```

Run `source setvars.sh` in every shell that builds or runs jitNN.

## jitNN

```bash
git clone https://github.com/beestream-ai/jitNN && cd jitNN
./mvnw install -DskipTests
./mvnw test                                          # reference backend only
./mvnw test -Paccel-tests -Djitnn.test.device=0:0    # plus the device tests
```

The Maven wrapper downloads Maven 3.9.6 on first use; `./mvnw -o` builds offline once the
dependencies are in `~/.m2`. To build against a TornadoVM `develop` install instead of the
release, pass its version on every Maven command: `-Dtornadovm.version=8.1.1-jdk22plus-dev`.

### Releases

Released versions are on the [GitHub releases page](https://github.com/beestream-ai/jitNN/releases)
(the module jars and the Javadoc) and in GitHub Packages. To depend on them from Maven, add the
repository to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>github-jitnn</id>
        <url>https://maven.pkg.github.com/beestream-ai/jitNN</url>
    </repository>
</repositories>

<dependency>
    <groupId>ai.beestream</groupId>
    <artifactId>jitnn-nn</artifactId>
    <version>0.1.0</version>
</dependency>
```

GitHub Packages asks for a login even for public packages: add a `<server>` with id `github-jitnn`,
your GitHub user name and a personal access token with the `read:packages` scope to
`~/.m2/settings.xml`.

Maintainers release from the Actions tab: Release > Run workflow, with the version (for example
`0.1.0`). It sets that version on a commit of its own, tags it `v0.1.0` (`main` keeps its
`-SNAPSHOT` version), runs the tests, publishes to GitHub Packages and creates the GitHub release.
"Dry run" stops after building and testing.

## Test data

The MNIST gate needs the four MNIST IDX files (gzipped is fine) and a PyTorch reference run in the
same directory:

```bash
mkdir -p ~/jitnn-data/mnist && cd ~/jitnn-data/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fLO https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
done
python <jitnn>/scripts/pytorch/mnist_mlp.py ~/jitnn-data/mnist ~/jitnn-data/mnist/mnist_mlp_pytorch.f32
./mvnw test -Paccel-tests -Djitnn.mnist=$HOME/jitnn-data/mnist
```

The CIFAR-10 gates need the binary CIFAR-10 (170 MB) and PyTorch references (set
`JITNN_TORCH_DEVICE=cuda` or `mps` to make them on a GPU):

```bash
mkdir -p ~/jitnn-data/cifar10 && cd ~/jitnn-data/cifar10
curl -fLO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz && tar xzf cifar-10-binary.tar.gz
for m in resnet18 mobilenetv2; do
  python <jitnn>/scripts/pytorch/cnn_train.py $m . ${m}_cifar10_pytorch.f32 0 cifar10
  python <jitnn>/scripts/pytorch/cnn_grad64.py $m . ${m}_cifar10_pytorch.f32 cifar10
done
./mvnw test -Paccel-tests -Djitnn.cifar10=$HOME/jitnn-data/cifar10 -Dtest='CnnAccelTest#*Cifar10*'
```

The transformer gates train on a byte-level corpus. Any text works; the documented results use
TornadoVM's documentation (`docs/source/*.rst` concatenated):

```bash
mkdir -p ~/jitnn-data/text && cat <TornadoVM>/docs/source/*.rst > ~/jitnn-data/text/corpus.txt
for m in gpt2 llama bert; do
  python <jitnn>/scripts/pytorch/transformers.py train  $m ~/jitnn-data/text/corpus.txt ~/jitnn-data/text/${m}_pytorch.f32 300
  python <jitnn>/scripts/pytorch/transformers.py grad64 $m ~/jitnn-data/text/corpus.txt ~/jitnn-data/text/${m}_pytorch.f32
done
./mvnw test -Paccel-tests -Djitnn.text=$HOME/jitnn-data/text -Dtest=TransformerAccelTest
```

The Hugging Face test downloads `gpt2` and `bert-base-uncased` and compares jitNN's tokenizers and
models with the `tokenizers` and `transformers` libraries:

```bash
pip install torch transformers tokenizers huggingface_hub safetensors
python <jitnn>/scripts/hf/hf_reference.py ~/jitnn-data/hf
./mvnw test -Paccel-tests -Djitnn.hf=$HOME/jitnn-data/hf -Dtest=HfAccelTest
```

## Regenerating the references

All references are committed; regenerate them only to change or extend them:

```bash
python3 -m venv ~/jitnn-data/torch-venv
~/jitnn-data/torch-venv/bin/pip install torch numpy scikit-learn onnx onnxruntime gguf
~/jitnn-data/torch-venv/bin/python scripts/pytorch/goldens.py jitnn-core/src/test/resources/golden
~/jitnn-data/torch-venv/bin/python scripts/sklearn/ml_goldens.py jitnn-nn/src/test/resources/ml
~/jitnn-data/torch-venv/bin/python scripts/onnx/onnx_goldens.py jitnn-onnx/src/test/resources/onnx
~/jitnn-data/torch-venv/bin/python scripts/gguf/gguf_fixture.py jitnn-gguf/src/test/resources/gguf/fixture.gguf
```

The TensorFlow goldens need TensorFlow (CPU is enough), best in a separate environment:

```bash
python3 -m venv ~/jitnn-data/tf-venv
~/jitnn-data/tf-venv/bin/pip install tensorflow numpy
~/jitnn-data/tf-venv/bin/python scripts/tf/tf_goldens.py jitnn-tf/src/test/resources/tf
```
