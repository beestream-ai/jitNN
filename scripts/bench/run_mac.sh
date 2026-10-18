#!/bin/bash
# ResNet-50 and BERT-base inference on an Apple GPU: jitNN on TornadoVM's Metal backend, once on its
# own kernels and once with MLX (--with-native), then PyTorch (MPS) and ONNX Runtime (CoreML provider,
# CPU and GPU only) on the same models and inputs. Run from the repository root.
#   scripts/bench/run_mac.sh <device> <out dir>
# <device> is the Metal device as jitNN's tests name it (backend:device, see `tornado --devices`).
# Needs:
#   - TORNADOVM_HOME set to a Metal build of TornadoVM develop, 7.3.2-dev or newer;
#   - `brew install mlx` for the native run;
#   - a Python (PYTHON=..., default python3) with torch, onnx, onnxscript and onnxruntime.
# Options through the environment: MODELS (default resnet50,bert), DEVICE_MEMORY (default 24GB),
# DIRECT_MEMORY (default 32g: Metal buffers are direct host memory, and the JVM's default limit, the
# heap size, is too small for the cached plans of every batch size),
# METAL_EXTRA (more JVM options for the jitNN runs), SKIP_JITNN=1 / SKIP_FRAMEWORKS=1.
set -e
DEV=${1:?usage: run_mac.sh <device> <out dir>}
OUT=${2:?usage: run_mac.sh <device> <out dir>}
MODELS=${MODELS:-resnet50,bert}
MVN=${MVN:-./mvnw}
PY=${PYTHON:-python3}
mkdir -p "$OUT"
echo "jitNN $(git rev-parse --short HEAD), TornadoVM $TORNADOVM_HOME, $(sysctl -n machdep.cpu.brand_string)" | tee "$OUT/setup.txt"

if [ "${SKIP_JITNN:-0}" != 1 ]; then
  $MVN -q install -DskipTests
  BENCH="-XX:MaxDirectMemorySize=${DIRECT_MEMORY:-32g} -Djitnn.bench.inference=true -Djitnn.bench.models=$MODELS ${METAL_EXTRA:-}"
  for mode in kernels native; do
    extra=$BENCH
    # MLX for the products, softmax, norms and attention where it is available; jitNN's kernels elsewhere
    [ $mode = native ] && extra="$extra -Djitnn.native=auto"
    echo "== jitNN on Metal, $mode"
    $MVN -q test -Paccel-tests -Djitnn.test.device="$DEV" -Djitnn.test.deviceMemory="${DEVICE_MEMORY:-24GB}" -pl jitnn-nn \
      -Dtest=InferenceBenchAccelTest "-Djitnn.test.extra=$extra" > "$OUT/jitnn-$mode.log" 2>&1 || true
    grep -E "\[t1\]|\[jitnn\] native|Tests run:|FAIL|Exception" "$OUT/jitnn-$mode.log" | grep -v "Time elapsed: .* s -- in" || true
  done
fi

if [ "${SKIP_FRAMEWORKS:-0}" != 1 ]; then
  echo "== PyTorch (MPS) and ONNX Runtime (CoreML)"
  $PY scripts/bench/t1.py --device mps --models "$MODELS" "$OUT/models" 2>&1 | tee "$OUT/frameworks.log" | grep "\[t1\]" || true
fi
