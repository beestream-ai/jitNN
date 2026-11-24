# Installing and building jitNN

## Requirements

| What | Version | Notes |
|---|---|---|
| JDK | 22 or newer | compiled with `--release 22`; tested on JDK 25 |
| TornadoVM | 8.2.0 (8.1.0 also runs, without the fused cuBLASLt and cuDNN epilogues) | needs `TornadoExecutionPlan.transferToDevice(Object...)`, `withDynamicShapes()` and the library tasks (`TaskGraph.libraryTask`) with their modules (`tornado-cublas`, `tornado-cudnn`, `tornado-cufft`, `tornado-cusparse`, `tornado-mlx`), all on Maven Central |
| A device | CUDA, OpenCL or Metal | optional: the reference backend runs on any CPU |
| CUDA toolkit | 12 or 13; 13 for Blackwell GPUs (RTX 50 series) | optional, for the CUDA backend |
| cuDNN, MLX | cuDNN 9; MLX from `brew install mlx` | optional, for `-Djitnn.native` (see [NATIVE.md](NATIVE.md)); cuBLAS, cuFFT and cuSPARSE come with the CUDA toolkit |
| Python packages | `torch`, `numpy`, `gguf` (llama.cpp's), `onnx`, `onnxruntime`: any recent; `scikit-learn` 1.9.1; `tensorflow` 2.x with Keras 3 | optional, only to regenerate the [references](#regenerating-the-references) |

jitNN depends only on TornadoVM's `tornado-api` and library modules, which Maven fetches from
Maven Central; the TornadoVM SDK provides them at run time. `jitnn-onnx` and `jitnn-tf` use their own protobuf reader, `jitnn-gguf` its own GGUF
reader. `jitnn-djl` depends on DJL's `ai.djl:api` (0.38.0). The optional `jitnn-llm`
(`./mvnw -Pllm ...`) takes jitllm from Maven Central; it must be built against jitNN's TornadoVM
(jitllm 3.0.0 and later: 8.1.0).

## TornadoVM

jitNN runs on a TornadoVM 8.2.0 SDK, the version it compiles against. Install it, and a JDK, with
[SDKMAN!](https://sdkman.io), which keeps several versions side by side:

```bash
curl -s "https://get.sdkman.io" | bash          # once, then open a new shell
sdk install java 25.0.4-tem                     # any JDK 22+; the TornadoVM SDK does not include one
sdk install tornadovm 8.2.0-jdk22plus-metal     # the SDK for your platform, from the table below
tornado --devices                               # device ids, as <driver>:<device>
```

| Platform | SDKMAN! identifier |
|---|---|
| macOS, Apple Silicon | `8.2.0-jdk22plus-metal` (recommended) or `8.2.0-jdk22plus-opencl` |
| Linux x86-64, NVIDIA GPU | `8.2.0-jdk22plus-cuda` |
| Linux x86-64, other OpenCL GPU | `8.2.0-jdk22plus-opencl` |
| Linux x86-64, every backend in one SDK | `8.2.0-jdk22plus-full` |
| Linux AArch64 (such as NVIDIA Grace), Windows | not on SDKMAN!: see below |

Use the `jdk22plus` SDKs; the `jdk21` ones do not run jitNN. Every SDK includes the library modules
jitNN uses (`tornado-cublas`, `tornado-cudnn`, `tornado-cufft`, `tornado-cusparse`, `tornado-mlx`).
SDKMAN! sets `TORNADOVM_HOME` and `PATH` for the default version, which
`sdk default tornadovm <identifier>` changes; `sdk use tornadovm <identifier>` switches the current
shell only, and `sdk list tornadovm` shows what is available.

### Without SDKMAN!

Every SDK, including CUDA for Linux AArch64 and Windows, is attached to the
[TornadoVM 8.2.0 release](https://github.com/beehive-lab/TornadoVM/releases/tag/v8.2.0) and on
[tornadovm.org/downloads](https://www.tornadovm.org/downloads). Unpack it and point `TORNADOVM_HOME`
at it:

```bash
curl -fLO https://github.com/beehive-lab/TornadoVM/releases/download/v8.2.0/tornadovm-8.2.0-jdk22plus-cuda-linux-aarch64.tar.gz
tar xzf tornadovm-8.2.0-jdk22plus-cuda-linux-aarch64.tar.gz
export TORNADOVM_HOME=$PWD/tornadovm-8.2.0-cuda PATH=$PWD/tornadovm-8.2.0-cuda/bin:$PATH
tornado --devices
```

### Building TornadoVM

Build from source only for TornadoVM's `develop` branch, with `JAVA_HOME` at the JDK you will run
jitNN with:

```bash
git clone -b develop https://github.com/beehive-lab/TornadoVM && cd TornadoVM
make jdk22plus BACKEND=opencl,metal      # or BACKEND=cuda, or several
source setvars.sh                        # JAVA_HOME, TORNADOVM_HOME, PATH
```

Run `source setvars.sh` in every shell that builds or runs jitNN, and pass the SDK's version on
every Maven command (below).

## jitNN

```bash
git clone https://github.com/beestream-ai/jitNN && cd jitNN
./mvnw install -DskipTests
./mvnw test                                          # reference backend only
./mvnw test -Paccel-tests -Djitnn.test.device=0:0    # plus the device tests
```

The Maven wrapper downloads Maven 3.9.6 on first use; `./mvnw -o` builds offline once the
dependencies are in `~/.m2`. To build against a TornadoVM `develop` install instead of the
release, pass its version on every Maven command: `-Dtornadovm.version=8.2.1-jdk22plus-dev`.

### Releases

Released versions are on the [GitHub releases page](https://github.com/beestream-ai/jitNN/releases).
Each release has one jar, `jitnn-<version>.jar`, with every module in it, and the Javadoc. Releases
published for Maven are also in GitHub Packages, one artifact per module.

#### Which module do I need

Depend on the module for what you do; Maven brings the jitNN modules it builds on:

| You want to | Depend on | It also brings |
|---|---|---|
| Tensors, autograd, operators, `Linalg`, `Fft`, data-parallel training | `jitnn-core` | |
| Layers, models, optimisers, data loaders, tokenizers, classical ML | `jitnn-nn` | core |
| Run or export ONNX models | `jitnn-onnx` | nn, core |
| Run TensorFlow SavedModels and GraphDefs | `jitnn-tf` | onnx, nn, core |
| Read GGUF quantised weights | `jitnn-gguf` | core |
| Use jitNN as DJL's engine | `jitnn-djl` | onnx, nn, core, DJL's `api` |

`jitnn-llm` (text generation through jitllm) is not released; build it with `-Pllm`. TornadoVM is a
`provided` dependency of every module: it comes from the TornadoVM SDK at run time
(`java @$TORNADOVM_HOME/tornado-argfile ...`), never from Maven.

#### Maven

Import `jitnn-bom` once, then list modules without versions, so they always match:

```xml
<repositories>
    <repository>
        <id>github-jitnn</id>
        <url>https://maven.pkg.github.com/beestream-ai/jitNN</url>
    </repository>
</repositories>

<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>ai.beestream</groupId>
            <artifactId>jitnn-bom</artifactId>
            <version>0.1.0-alpha</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<dependencies>
    <dependency>
        <groupId>ai.beestream</groupId>
        <artifactId>jitnn-onnx</artifactId>
    </dependency>
</dependencies>
```

GitHub Packages asks for a login even for public packages: add a `<server>` with id `github-jitnn`,
your GitHub user name and a personal access token with the `read:packages` scope to
`~/.m2/settings.xml`.

#### The one jar, without a build tool

`jitnn-<version>.jar` holds every module except `jitnn-llm`, and nothing else: TornadoVM comes
from its SDK, and DJL (only for `jitnn-djl`) from your application. Take it from the release page,
or build it from a checkout:

```bash
./mvnw install -DskipTests       # writes jitnn-all/target/jitnn-<version>.jar
```

The commands below use `jitnn-0.1.0-alpha.jar` and a program `Hello.java`:

```java
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;

public class Hello {
    public static void main(String[] args) {
        Tensor a = Tensor.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        Tensor b = Tensor.of(new float[] { 1, 0, 0, 1, 1, 1 }, 3, 2);
        System.out.println(JitnnRuntime.backend() + ": " + java.util.Arrays.toString(a.matmul(b).toFloatArray()));
    }
}
```

**Compile.** `javac` also needs TornadoVM's API jar, from the SDK, because some `Tensor` methods
take TornadoVM array types:

```bash
javac -cp jitnn-0.1.0-alpha.jar:$TORNADOVM_HOME/share/java/tornado/tornado-api-8.2.0.jar Hello.java
```

**Run on a GPU.** The TornadoVM argument file sets up its JVM; `-Djitnn.device=<driver>:<device>`
picks a device (`tornado --devices` lists them):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp jitnn-0.1.0-alpha.jar:. Hello
# TORNADO: [4.0, 5.0, 10.0, 11.0]
```

The argument file puts the current directory on the module path, so run from a directory without
unrelated or damaged jars in it (macOS `._*.jar` files left by copying count too).
`java @$TORNADOVM_HOME/tornado-argfile -cp jitnn-0.1.0-alpha.jar Hello.java` compiles and runs
in one step.

**Run on the CPU, without TornadoVM.** Any JDK 22 or newer; the TornadoVM API jar must be on the
class path (from the SDK, or `tornado-api-8.2.0-jdk22plus.jar` from
[Maven Central](https://repo1.maven.org/maven2/io/github/beehive-lab/tornado-api/8.2.0-jdk22plus/)):

```bash
java --add-modules jdk.incubator.vector -Djitnn.backend=cpu -cp jitnn-0.1.0-alpha.jar:tornado-api-8.2.0-jdk22plus.jar:. Hello
# CPU: [4.0, 5.0, 10.0, 11.0]
```

Without `--add-modules jdk.incubator.vector`, jitNN uses the `reference` backend (scalar Java).

#### Making a release

Maintainers release from the Actions tab: Release > Run workflow, with the version (for example
`0.1.0-alpha`). It sets that version on a commit of its own and tags it `v0.1.0-alpha` (`main` keeps its
`-SNAPSHOT` version), runs the tests, publishes the modules and the BOM to GitHub Packages and
creates the GitHub release. "Dry run" stops after building and testing.

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

`scripts/ci/stage-test-data.sh [data dir] [mnist cifar10 text hf llm]` does all of the above, and also
makes the references of the LeNet, CNN-on-MNIST, VAE, DCGAN, U-Net, LSTM, GRU and Llama-family gates.
It puts everything in `~/jitnn-data` by default, with its own PyTorch venv there. Set
`JITNN_TORCH_DEVICE=cuda` or `mps` to compute the references on a GPU.

## Continuous integration

`.github/workflows/pr-tests.yml` builds and tests every pull request to `main` on two self-hosted
runners. Each TornadoVM backend gets its own job, in two chains that run in parallel:

| Runner labels | First job | Then |
|---|---|---|
| `self-hosted, Linux, X64` | `linux-x64 / CUDA`: build, host tests, CUDA | `linux-x64 / OpenCL`: build, OpenCL |
| `self-hosted, macOS, ARM64` | `macos-arm64 / Metal`: build, host tests, Metal | `macos-arm64 / OpenCL`: build, OpenCL |

Every build includes `jitnn-llm` (`-Pllm`). Pull requests that change no code skip the workflow: those
that touch only Markdown files, `docs/`, `.readthedocs.yaml`, `LICENSE`, `NOTICE`, `.gitignore` or the
Javadoc, release and jitllm bump workflows.
`docs/CONFIGURATION.md` still triggers it, since `ConfigurationDocTest` checks its generated tables
against the code. The workflow can always be run by hand (Actions > PR tests > Run workflow).

The second job of a chain starts when the first ends, even if the first failed. In each job,
`scripts/ci/accel-tests.sh <backend>` runs `./mvnw test -Paccel-tests` on the backend's device. It
finds the SDK, writes its `tornado-argfile` if the SDK has never been run, and prints
`tornado --devices`. It passes the data gates whose data it finds under `~/jitnn-data`, with a
warning for each gate it skips. Each job uploads its reports (`ci-reports/<backend>`). Jobs run only
for branches of this repository, not for forks, because self-hosted runners execute the pull
request's code. Draft pull requests are skipped until they are marked ready. The script also runs
locally, after `./mvnw install`.

Each runner needs a JDK 22+ and the TornadoVM 8.2.0 SDKs installed with SDKMAN!
(`8.2.0-jdk22plus-cuda` and `-opencl` on Linux, `-metal` and `-opencl` on macOS). Run
`scripts/ci/stage-test-data.sh` once on each runner. Repository variables override the defaults:

| Variable | Default |
|---|---|
| `JITNN_JAVA_HOME` | `~/.sdkman/candidates/java/current` |
| `TORNADOVM_CUDA_HOME`, `TORNADOVM_OPENCL_LINUX_HOME`, `TORNADOVM_METAL_HOME`, `TORNADOVM_OPENCL_MAC_HOME` | `~/.sdkman/candidates/tornadovm/8.2.0-jdk22plus-<backend>` |
| `JITNN_DEVICE_CUDA`, `JITNN_DEVICE_OPENCL_LINUX`, `JITNN_DEVICE_METAL`, `JITNN_DEVICE_OPENCL_MAC` | `0:0` |
| `JITNN_DATA` | `~/jitnn-data` |
| `JITNN_DEVICE_MEMORY` | `12GB` |

`.github/workflows/jitllm-bump.yml` opens a pull request for each new jitllm release on Maven Central;
its header describes the setup.

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
