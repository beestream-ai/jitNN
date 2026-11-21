# Native image

A jitNN application can be compiled ahead of time into a GraalVM native image that runs its
operators on TornadoVM's CUDA backend. The image computes the same results as the JVM, bit for
bit, starts sooner and uses much less memory; a JVM that has warmed up is faster per call.

## Requirements

- GraalVM for JDK 25 with `native-image` (tested with Oracle GraalVM 25.0.4).
- A TornadoVM jdk22plus SDK in `TORNADOVM_HOME`, built from a TornadoVM that does not use shared
  arenas in a native image. The image builder cannot compile the cpu backend's Vector API code
  with `-H:+SharedArenaSupport`, and refuses Vector API support with shared arenas; TornadoVM's
  CUDA driver allocates its pinned host buffers and cuBLASLt plans from shared arenas today.
- Recommended: a TornadoVM with bulk copies in `fromArray` and `toHeapArray`
  ([beehive-lab/TornadoVM#1342](https://github.com/beehive-lab/TornadoVM/pull/1342)). Without
  them, loading a model in an image copies every weight element by element: ResNet-50 takes
  1.30 s to load instead of 0.52 s.

## Building an image

Add `jitnn-native-image` to the application's class path. Its jar carries the builder flags and a
GraalVM `Feature` that registers jitNN's kernels, so `native-image` needs no tracing run. The
script in the module builds an image of any main class against the SDK in `TORNADOVM_HOME`:

```bash
./mvnw install -DskipTests -Dtornadovm.version=<your TornadoVM version>
jitnn-native-image/native-image.sh build MnistMlp
```

A bare class name is looked up in `ai.beestream.jitnn.examples`; give a full name for your own
program (`APP_CP` sets its class path). Building takes about a minute and 21 GB of memory on 32
threads; the image is about 90 MB.

Run the image with `TORNADOVM_HOME` exported (TornadoVM keeps its CUDA code cache in
`$TORNADOVM_HOME/var`) and the library path the `tornado` launcher would set:

```bash
jitnn-native-image/target/MnistMlp -Djava.library.path=$TORNADOVM_HOME/lib -Dtornado.device.memory=12GB 3
```

`native-image.sh java <main>` runs the same program on the JVM with TornadoVM's argfile, to compare.

## What the module holds

| File | Role |
|---|---|
| `JitnnFeature` | registers every jitNN class and its declared methods for reflection (TornadoVM resolves a kernel with `getDeclaredMethod`), the classes with serializable task lambdas as lambda-capturing, and the TornadoVM API types kernels use |
| `META-INF/native-image/ai.beestream/jitnn/` | the builder flags (`-H:+ForeignAPISupport`, `-H:+VectorAPISupport`) and jitNN's class files and tuning tables as resources |
| `tornadovm/reachability-metadata.json` | what TornadoVM and the JDK reach by reflection and FFM downcalls (cuBLAS included); belongs in TornadoVM's own jars once it ships native-image metadata |
| `examples/reachability-metadata.json` | the ImageIO and Java 2D entries the image-classification examples need |
| `native-image.sh` | `build`, `trace` (records metadata with GraalVM's tracing agent) and `java` |

A program that reaches more of TornadoVM or the JDK than the examples do (another vendor library,
another file format) may need more metadata: run it once with `native-image.sh trace <main>`, and
add the result under `target/trace/` as another configuration directory.

## Results

MNIST MLP training (jitnn-examples MnistMlp) and ONNX ResNet-50 inference (OnnxImageClassification)
on an RTX 5070 Ti, CUDA 13.0, medians of 5 launches with a warm code cache:

| | JDK 25 | JDK 25 + Leyden AOT cache | Native image |
|---|---|---|---|
| MLP, launch to first step done | 1.27 s | 1.20 s | 0.71 s |
| ResNet-50, launch to first result | 2.22 s | 2.08 s | 1.61 s |
| Peak memory, MLP / ResNet-50 | 1.8 / 2.5 GB | 1.8 / 2.5 GB | 0.54 / 1.0 GB |
| MLP step after 15 epochs | 0.115 ms | 0.113 ms | 0.163 ms |
| ResNet-50 call, median of 1,000 | 1.51 ms | 1.51 ms | 1.61 ms |

Losses, accuracies, logits and top-5 were bit-identical to the JVM's in every run, with jitNN's
kernels and with cuBLAS (`-Djitnn.native=auto`). The image suits short-lived jobs and services
that scale out; for long training runs the JVM is faster per step.

## Limits

- The `cpu` backend does not run in an image yet: with `-H:+VectorAPISupport` its GEMM kernel
  crashes in the image builder's Vector API code, and without it the portable Vector API is about
  90 times slower than on the JVM. The `tornado` and `reference` backends work.
- Tested on Linux x86-64 with CUDA only.
