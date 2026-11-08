# API reference (Javadoc)

The Javadoc of every jitNN module (`jitnn-core`, `jitnn-nn`, `jitnn-onnx`, `jitnn-gguf`, `jitnn-tf`,
`jitnn-djl`) is published with this site:

<p><a class="reference external" href="api/index.html"><strong>Open the jitNN API reference</strong></a></p>

The packages, by area:

| Package | Contents |
|---|---|
| `ai.beestream.jitnn` | `Tensor`, autograd, `F` (functional operators), `Linalg`, `Fft`, `SparseMatrix`, `Autocast`, `CustomOp` |
| `ai.beestream.jitnn.engine` | lazy execution: traces, compiled plans, devices, autotuning, vendor libraries |
| `ai.beestream.jitnn.ops`, `ai.beestream.jitnn.kernels` | the operators and their TornadoVM kernels |
| `ai.beestream.jitnn.dist` | collectives and ranks across devices |
| `ai.beestream.jitnn.io` | checkpoints in the safetensors format |
| `ai.beestream.jitnn.nn`, `.nn.models`, `.nn.optim` | modules and layers, models, optimisers |
| `ai.beestream.jitnn.ml` | classical machine learning |
| `ai.beestream.jitnn.data` | datasets |
| `ai.beestream.jitnn.onnx`, `.tf`, `.gguf`, `.djl` | the ONNX and TensorFlow importers, the GGUF reader, the DJL engine |

## Building it locally

With JDK 22+ and the TornadoVM jars in the local Maven repository (see [INSTALL.md](INSTALL.md)):

```bash
./mvnw install -DskipTests
./mvnw javadoc:aggregate        # writes target/reports/apidocs/index.html
```

Without them, `docs/build-javadoc.sh <output directory>` downloads a JDK and builds TornadoVM's Java
modules first, as the Read the Docs build does. `make -C docs html-with-api` builds this site with
the Javadoc under `_build/html/api`.
