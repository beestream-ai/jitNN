#!/bin/bash
# The kernel benchmark on one device: three JVM runs of jitnn-bench (device times), interleaved with
# the PyTorch baselines, a fourth run without the profiler (host times), then the summary. Run from the repository root on an idle GPU.
#   scripts/bench/run_kernels.sh <device, e.g. 1:0> <out dir>
# Environment: MVN (Maven command, default ./mvnw), PYTHON, KBENCH_EXTRA (more -D properties, e.g.
# -Djitnn.kbench.workloads=mlp_train_b256 or -Djitnn.kbench.families=gemm,convolution), NO_BASELINE=1.
set -e
DEV=$1; OUT=$2; shift 2
mkdir -p "$OUT"
LABEL="jitnn-$(git rev-parse --short HEAD)"
MVN=${MVN:-./mvnw}
PY=${PYTHON:-python}
run() {
  PROF=${2:-true}
  $MVN -q test -Paccel-tests -Djitnn.test.device=$DEV -Djitnn.test.profile=$PROF -pl jitnn-bench -Dtest=KernelBenchAccelTest \
    "-Djitnn.test.extra=-Djitnn.kbench=true -Djitnn.cudaGraph=false -Djitnn.kbench.out=$OUT/run$1.jsonl -Djitnn.kbench.label=$LABEL $KBENCH_EXTRA" \
    2>&1 | grep -E "\[kbench\]|Tests run:|FAIL|ERROR" | grep -v "^\[kbench\] *[0-9]*/" || true
}
run 1
[ -n "$NO_BASELINE" ] || $PY scripts/bench/kernels.py "$OUT/run1.jsonl" "$OUT/baselines.jsonl"
run 2
run 3
run host false
$PY scripts/bench/compare.py summary "$OUT/run[0-9].jsonl" --host "$OUT/runhost.jsonl" $( [ -f "$OUT/baselines.jsonl" ] && echo --baselines "$OUT/baselines.jsonl" )
