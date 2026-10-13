#!/bin/bash
# The end-to-end model profiles for step 2: one JVM with the profiler (per-operator device time),
# one without it and with CUDA graphs as normal (wall time). Run from the repository root.
#   scripts/bench/run_models.sh <device> <out dir> [--with-native]
set -e
DEV=$1; OUT=$2
# --with-native as a third argument: jitNN calls cuBLAS, cuDNN, ... (or MLX) where available
if [ "${3:-}" = "--with-native" ]; then KBENCH_EXTRA="$KBENCH_EXTRA -Djitnn.native=auto"; fi
mkdir -p "$OUT"
MVN=${MVN:-./mvnw}
LABEL="jitnn-$(git rev-parse --short HEAD)"
$MVN -q test -Paccel-tests -Djitnn.test.device=$DEV -Djitnn.test.profile=true -Djitnn.test.deviceMemory=40GB -pl jitnn-bench -Dtest=ModelProfileAccelTest \
  "-Djitnn.test.extra=-Djitnn.models=true -Djitnn.cudaGraph=false -Djitnn.kbench.out=$OUT/models_profiled.jsonl -Djitnn.kbench.label=$LABEL $KBENCH_EXTRA" 2>&1 | grep -E "\[models\]|Tests run:|FAIL|ERROR" | head -40 || true
$MVN -q test -Paccel-tests -Djitnn.test.device=$DEV -Djitnn.test.deviceMemory=40GB -pl jitnn-bench -Dtest=ModelProfileAccelTest \
  "-Djitnn.test.extra=-Djitnn.models=true -Djitnn.kbench.out=$OUT/models_wall.jsonl -Djitnn.kbench.label=$LABEL $KBENCH_EXTRA" 2>&1 | grep -E "\[models\]|Tests run:|FAIL|ERROR" | head -40 || true
