#!/bin/bash
# The GPU head-to-head: jitNN against PyTorch, ONNX Runtime and TensorRT on ResNet-50 and BERT-base
# inference (strict FP32, TF32, FP16) and three training workloads, each framework in the fastest
# variant found by sweeping gsweep.py / gtrain.py, in interleaved rounds with the framework order rotated.
# Records host-to-host times, first-call times and peak GPU memory (nvidia-smi every 100 ms, so run
# it on a GPU nothing else uses). Run from the repository root, then summarise:
#   scripts/bench/gpu_final.sh <out dir>
#   python scripts/bench/gpu_summary.py <out dir>/final.txt
# Needs, through the environment:
#   - TORNADOVM_HOME: a CUDA build of TornadoVM develop, 8.0.1-dev or newer, and cuDNN on LD_LIBRARY_PATH;
#   - PYTHON: a Python with torch, torch_tensorrt, onnx and onnxscript;
#   - PYTHON_ORT: a Python with onnxruntime-gpu whose TensorRT provider finds its TensorRT (ORT 1.31
#     takes TensorRT 10, Torch-TensorRT 2.14 takes 11: two environments), with ORT_LIBRARY_PATH the
#     library path it needs (its tensorrt_libs and nvidia/*/lib directories);
#   - optional: DEVICE (jitNN's test device, default 0:0), ROUNDS (default 3), DEVICE_MEMORY (default 20GB).
# The Python runs get only their own libraries: a cuDNN on LD_LIBRARY_PATH for jitNN breaks PyTorch's.
set -u
OUT=${1:?usage: gpu_final.sh <out dir>}
PY=${PYTHON:?set PYTHON}; PYO=${PYTHON_ORT:?set PYTHON_ORT}; ORTLD=${ORT_LIBRARY_PATH:-}
DEV=${DEVICE:-0:0}; ROUNDS=${ROUNDS:-3}; DMEM=${DEVICE_MEMORY:-20GB}
T=$(pwd)/scripts/bench; W=$OUT/work; MVN=${MVN:-./mvnw}
export TORCHINDUCTOR_CACHE_DIR=$W/inductor
mkdir -p $OUT/logs $W
STRICT=NVIDIA_TF32_OVERRIDE=0
NONE=JITNN_BENCH=1

# peak <tag> <cmd...>: runs the command, appends "[mem] <tag> <peak MB> <idle MB>"
peak() {
  local tag=$1; shift
  local base=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > $OUT/mem.tmp &
  local m=$!
  "$@"
  kill $m; wait $m 2>/dev/null
  echo "[mem] $tag $(sort -n $OUT/mem.tmp | tail -1) $base" >> $OUT/final.txt
}

gs() { # round python library-path framework variant kind batch seq precision
  local r=$1 py=$2 ld=$3; shift 3
  local tag=r$r-$1-$2-$3-$4-$5-$6
  peak $tag env LD_LIBRARY_PATH="$ld" timeout 2400 $py $T/gsweep.py $T $W "$@" > $OUT/logs/$tag.log 2>&1
  grep "^\[gsweep\]" $OUT/logs/$tag.log | sed "s/^/r$r /" >> $OUT/final.txt || echo "r$r [gsweep] $* FAILED" >> $OUT/final.txt
}

jinf() { # round name env only extra...
  local r=$1 name=$2 envs=$3 only=$4; shift 4
  local tag=r$r-jitnn-$name-${only:-all}
  peak $tag bash -c "env $envs timeout 7200 $MVN -q test -Paccel-tests -Djitnn.test.device=$DEV -Djitnn.test.deviceMemory=$DMEM -pl jitnn-nn -Dtest=InferenceBenchAccelTest \"-Djitnn.test.extra=-Djitnn.bench.inference=true -Djitnn.bench.only=$only $*\"" > $OUT/logs/$tag.log 2>&1
  grep -E "^\[t1\] jitnn" $OUT/logs/$tag.log | sed "s/^/r$r $name /" >> $OUT/final.txt
}

jtr() { # round name env extra...
  local r=$1 name=$2 envs=$3; shift 3
  local tag=r$r-jtrain-$name
  peak $tag bash -c "env $envs timeout 7200 $MVN -q test -Paccel-tests -Djitnn.test.device=$DEV -Djitnn.test.deviceMemory=$DMEM -pl jitnn-bench -Dtest=ModelProfileAccelTest \"-Djitnn.test.extra=-Djitnn.models=true -Djitnn.kbench.workloads=resnet50_cifar_train_b32,bert_train_b8_s128,gpt2_train_b8_s256 -Djitnn.kbench.out=$OUT/logs/$tag.jsonl $*\"" > $OUT/logs/$tag.log 2>&1
  grep -E "^\[model\]" $OUT/logs/$tag.log | sed "s/^/r$r $name /" >> $OUT/final.txt
}

ptr() { # round variant workload precision
  local tag=r$1-ptrain-$2-$3-$4
  peak $tag env LD_LIBRARY_PATH= timeout 2400 $PY $T/gtrain.py $T $W $2 $3 $4 > $OUT/logs/$tag.log 2>&1
  grep "^\[gtrain\]" $OUT/logs/$tag.log | sed "s/^/r$1 /" >> $OUT/final.txt || echo "r$1 [gtrain] $2 $3 $4 FAILED" >> $OUT/final.txt
}

$MVN -q install -DskipTests || exit 1
TF="-Djitnn.native=auto -Djitnn.native.tf32=true -Djitnn.native.convMinFlops=67108864"
# the fastest torch.compile mode per row and precision, and ONNX Runtime's fastest variant (from the sweep on an A10)
ptv() { case "$1:$2:$3" in
  resnet50:32:fp32|resnet50:128:fp32|bert:32:tf32|resnet50:32:fp16|resnet50:128:fp16|bert:1:fp16|bert:32:fp16) echo compile;;
  *) echo compile-ro;; esac; }
ortv() { if [ $1 = resnet50 ]; then echo exh-graph; elif [ "$1:$2:$3" = bert:8:fp16 ]; then echo opt-exh; else echo opt-exh-graph; fi; }
order() { case $(( $1 % 3 )) in 1) echo pt ort trt ortt;; 2) echo ort trt ortt pt;; *) echo trt ortt pt ort;; esac; }
SHAPES="resnet50:1:0 resnet50:32:0 resnet50:128:0 bert:1:128 bert:32:128 bert:8:512"

for r in $(seq 1 $ROUNDS); do
  # jitNN: one JVM per row in round 1 (peak memory per row), one JVM per configuration after
  if [ $r = 1 ]; then
    for s in $SHAPES; do IFS=: read k b t <<< "$s"
      n=$([ $k = resnet50 ] && echo resnet50_b$b || echo bert_b${b}_s$t)
      # strict FP32: jitNN's own kernels for ResNet, the vendor libraries (TF32 off) for BERT
      if [ $k = resnet50 ]; then jinf $r strict $STRICT ${n}_fp32; else jinf $r strict $STRICT ${n}_fp32 -Djitnn.native=auto; fi
      jinf $r tf32 $NONE ${n}_fp32 $TF
      jinf $r fp16 $NONE ${n}_fp16h $TF
    done
  else
    jinf $r strict-k $STRICT ""
    jinf $r strict-n $STRICT "" -Djitnn.native=auto
    jinf $r tf32 $NONE "" $TF
  fi
  for s in $SHAPES; do IFS=: read k b t <<< "$s"
    for p in fp32 tf32 fp16; do
      for f in $(order $r); do
        case $f in
          pt) gs $r $PY "" pytorch $(ptv $k $b $p) $k $b $t $p;;
          ort) gs $r $PYO "$ORTLD" ort $(ortv $k $b $p) $k $b $t $p;;
          trt) gs $r $PY "" trt dynamo $k $b $t $p;;
          # TensorRT's own FP32 is TF32: no strict FP32 row
          ortt) [ $p != fp32 ] && gs $r $PYO "$ORTLD" ort trt $k $b $t $p;;
        esac
      done
    done
  done
  jtr $r strict-k $STRICT
  jtr $r strict-n $STRICT -Djitnn.native=auto
  jtr $r tf32 $NONE -Djitnn.native=auto -Djitnn.native.tf32=true
  ptr $r eager resnet50_cifar_train_b32 fp32
  ptr $r compile-ro resnet50_cifar_train_b32 tf32
  ptr $r compile-ma bert_train_b8_s128 fp32
  ptr $r compile-ro bert_train_b8_s128 tf32
  ptr $r compile-ma gpt2_train_b8_s256 fp32
  ptr $r compile-ma gpt2_train_b8_s256 tf32
done
