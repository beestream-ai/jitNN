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
# usage: jitnn-examples/run.sh <example> [arguments...]
#   e.g. jitnn-examples/run.sh MnistMlp 3, or jitnn-examples/run.sh djl.TrainMnistDjl
# Runs one example on the TornadoVM SDK in TORNADOVM_HOME (source TornadoVM's setvars.sh first),
# after ./mvnw install -DskipTests in the jitNN checkout. JAVA_OPTS adds JVM flags, such as
# -Djitnn.device=1:0 (another device) or -Djitnn.verbose=true (log each compiled plan).
set -e
if [ $# -lt 1 ]; then
    sed -n '/^# usage:/,/^set -e/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    echo "examples:"
    ls "$(dirname "$0")"/src/main/java/ai/beestream/jitnn/examples/*.java "$(dirname "$0")"/src/main/java/ai/beestream/jitnn/examples/djl/*.java \
        | sed -e 's#.*/examples/##' -e 's#\.java$##' -e 's#/#.#' | grep -v '^Examples$' | sed 's/^/  /'
    exit 1
fi
[ -n "$TORNADOVM_HOME" ] || { echo "TORNADOVM_HOME is not set: source TornadoVM's setvars.sh first"; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)
cpfile=$here/target/classpath.txt
if [ ! -f "$cpfile" ] || [ "$here/pom.xml" -nt "$cpfile" ]; then
    (cd "$here/.." && ./mvnw -q -pl jitnn-examples dependency:build-classpath -Dmdep.outputFile="$cpfile" ${MAVEN_ARGS:-})
fi
example=$1
shift
exec java @"$TORNADOVM_HOME"/tornado-argfile --add-modules jdk.incubator.vector \
    -Dtornado.device.memory=${JITNN_DEVICE_MEMORY:-12GB} ${JAVA_OPTS:-} \
    -cp "$here/target/classes:$(cat "$cpfile")" ai.beestream.jitnn.examples."$example" "$@"
