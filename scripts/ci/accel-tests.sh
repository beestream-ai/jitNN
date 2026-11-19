#!/bin/bash
# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# usage: scripts/ci/accel-tests.sh <backend>
# Runs the accelerator tests (-Paccel-tests) on one TornadoVM backend (cuda, opencl or metal), as
# the PR workflow does on each self-hosted runner, after ./mvnw install. Run from the checkout root.
# It finds the backend's SDK, writes its tornado-argfile if the SDK has never been run, passes the
# data gates whose data is staged (scripts/ci/stage-test-data.sh), runs every module even when an
# earlier one has failures, copies the reports to ci-reports/<backend> and fails if any test failed.
# Environment (empty means the default):
#   JAVA_HOME              the JDK (default: SDKMAN!'s current java)
#   SDK_OVERRIDE           the SDK (default: SDKMAN!'s 8.1.0-jdk22plus-<backend>, else its newest 8.1.x -<backend>)
#   DEVICE_OVERRIDE        the device as <driver>:<device> (default 0:0)
#   DATA_OVERRIDE          the staged data (default ~/jitnn-data)
#   MEMORY_OVERRIDE        TornadoVM's device-memory budget (default 12GB)
#   SDK_VAR                the repository variable named in error messages
set -euo pipefail
backend=${1:?usage: accel-tests.sh <cuda|opencl|metal>}
candidates=${SDKMAN_DIR:-$HOME/.sdkman}/candidates
java_home=${JAVA_HOME:-$candidates/java/current}
[ -x "$java_home/bin/java" ] || { echo "::error::no JDK at $java_home (set JITNN_JAVA_HOME)"; exit 1; }

# an SDK ships tornado-argfile.template; its launcher writes tornado-argfile on first run
is_sdk() { [ -x "$1/bin/tornado" ] && { [ -f "$1/tornado-argfile" ] || [ -f "$1/tornado-argfile.template" ]; }; }
sdk=${SDK_OVERRIDE:-}
if [ -z "$sdk" ]; then
    sdk=$candidates/tornadovm/8.1.0-jdk22plus-$backend
    if ! is_sdk "$sdk"; then
        sdk=$(ls -d "$candidates"/tornadovm/8.1.*-"$backend" 2>/dev/null | sort | tail -1 || true)
    fi
fi
export JAVA_HOME=$java_home TORNADOVM_HOME=$sdk PATH=$java_home/bin:$sdk/bin:$PATH
if [ -n "$sdk" ] && is_sdk "$sdk" && [ ! -f "$sdk/tornado-argfile" ]; then
    echo "writing $sdk/tornado-argfile from its template"
    tornado --version >/dev/null
fi
if [ -z "$sdk" ] || [ ! -f "$sdk/tornado-argfile" ]; then
    echo "::error::no TornadoVM 8.1 $backend SDK on $(hostname) as $(whoami) (looked in $candidates/tornadovm; set ${SDK_VAR:-the SDK variable})"
    echo "TornadoVM SDKs in $candidates/tornadovm:"; ls -1 "$candidates/tornadovm" 2>&1 || true
    exit 1
fi
device=${DEVICE_OVERRIDE:-0:0}
data=${DATA_OVERRIDE:-$HOME/jitnn-data}
echo "JDK $java_home, TornadoVM $sdk, device $device"
tornado --version
tornado --devices

gates=()
gate() {  # <property> <dir under the data dir> <file that must exist there>
    if [ -e "$data/$2/$3" ]; then
        gates+=("-Djitnn.$1=$data/$2")
    else
        echo "::warning::$backend: $1 gate skipped, $data/$2/$3 not found (run scripts/ci/stage-test-data.sh on this runner)"
    fi
}
gate mnist   mnist   mnist_mlp_pytorch.f32
gate cifar10 cifar10 resnet18_cifar10_pytorch.f32
gate text    text    corpus.txt
gate hf      hf      reference.json

rm -rf ./*/target/surefire-reports "ci-reports/$backend"
# every module runs even when an earlier one has failures; the reports decide
./mvnw -B -ntp test -Paccel-tests -Dmaven.test.failure.ignore=true \
    -Djitnn.test.device="$device" \
    -Djitnn.test.deviceMemory="${MEMORY_OVERRIDE:-12GB}" \
    ${gates[@]+"${gates[@]}"}

mkdir -p "ci-reports/$backend"
cp ./*/target/surefire-reports/TEST-*.xml "ci-reports/$backend/" 2>/dev/null || { echo "::error::$backend: no test reports"; exit 1; }
failed=$(grep -lE '(failures|errors)="[1-9]' "ci-reports/$backend"/TEST-*.xml || true)
if [ -n "$failed" ]; then
    for f in $failed; do echo "::error::$backend: failures in $(basename "$f" .xml | sed 's/^TEST-//')"; done
    exit 1
fi
echo "$backend: $(ls "ci-reports/$backend" | wc -l | tr -d ' ') test classes, no failures"
