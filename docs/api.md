# API reference (Javadoc)

The Javadoc of jitNN's public API, every module (`jitnn-core`, `jitnn-nn`, `jitnn-onnx`, `jitnn-gguf`,
`jitnn-tf`, `jitnn-djl`), is published with this site:

<p><a class="reference external" href="api/index.html"><strong>Open the jitNN API reference</strong></a></p>

The packages, by area:

| Package | Contents |
|---|---|
| `ai.beestream.jitnn` | `Tensor`, autograd (`Grad`), `F` (functional operators), `Autocast`, `Linalg`, `Fft`, `SparseMatrix`, `CustomOp`, `JitnnRuntime`, `JitnnConfig`, `JitnnException` |
| `ai.beestream.jitnn.dist` | collectives and data-parallel training across devices and processes |
| `ai.beestream.jitnn.io` | checkpoints: state dicts, safetensors, Hugging Face |
| `ai.beestream.jitnn.engine` | only `Emitter` and `Node`, the types a `CustomOp` works with |
| `ai.beestream.jitnn.nn`, `.nn.models`, `.nn.optim`, `.nn.text` | modules and layers, models, optimisers, tokenizers |
| `ai.beestream.jitnn.ml` | classical machine learning |
| `ai.beestream.jitnn.data` | datasets and data loaders |
| `ai.beestream.jitnn.onnx`, `.tf`, `.gguf`, `.djl` | the ONNX and TensorFlow importers, the GGUF reader, the DJL engine |

The rest of `engine` and the `ops`, `kernels`, `cpu` and `natives` packages are internal and may
change in any release; they are left out of the API reference ([DEVELOPMENT.md](DEVELOPMENT.md),
"API and internals").

## Building it locally

With JDK 22+ (see [INSTALL.md](INSTALL.md)); Maven fetches the TornadoVM jars from Maven Central:

```bash
./mvnw install -DskipTests
./mvnw javadoc:aggregate                        # the public API, in target/reports/apidocs
./mvnw javadoc:aggregate -Pjavadoc-internals    # every package, the internal ones included
```

`docs/build-javadoc.sh <output directory>` does the same and downloads a JDK when none is found,
as the Read the Docs build does. `make -C docs html-with-api` builds this site with the Javadoc
under `_build/html/api`. The `Javadoc` GitHub workflow builds it on every push to `main` and
publishes it with GitHub Pages once Pages is enabled for the repository.
