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
# usage: jitnn-native-image/native-image.sh build <main class> [-o <output>] [native-image options...]
#        jitnn-native-image/native-image.sh trace <main class> [program arguments...]
#        jitnn-native-image/native-image.sh java  <main class> [program arguments...]
# Builds a GraalVM native image of a jitNN application on the TornadoVM SDK in TORNADOVM_HOME, or
# runs it on the JVM (java), or runs it under the tracing agent (trace) to record what TornadoVM and
# the JDK reach by reflection, JNI, resources and FFM downcalls into
# jitnn-native-image/target/trace/<main class>, to merge into tornadovm/reachability-metadata.json.
#   TORNADOVM_HOME  a jdk22plus TornadoVM SDK
#   GRAALVM_HOME    GraalVM for JDK 25 with native-image
#   APP_CP          the application's class path (default: jitnn-examples' classes and dependencies,
#                   after ./mvnw install -DskipTests)
#   JAVA_OPTS       JVM flags for java and trace
# Run the image with TORNADOVM_HOME exported (TornadoVM keeps its code cache in $TORNADOVM_HOME/var)
# and the library path TornadoVM's launcher would set:
#   ./MnistMlp -Djava.library.path=$TORNADOVM_HOME/lib -Dtornado.device.memory=12GB 3
set -euo pipefail
: "${TORNADOVM_HOME:?set TORNADOVM_HOME to a TornadoVM jdk22plus SDK}"
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
cmd=${1:?build, trace or java}
main=${2:?main class}
shift 2
[[ $main == *.* ]] || main=ai.beestream.jitnn.examples.$main

if [ -z "${APP_CP:-}" ]; then
    cpfile=$root/jitnn-examples/target/classpath.txt
    if [ ! -f "$cpfile" ] || [ "$root/jitnn-examples/pom.xml" -nt "$cpfile" ]; then
        (cd "$root" && ./mvnw -q -pl jitnn-examples dependency:build-classpath -Dmdep.outputFile="$cpfile" ${MAVEN_ARGS:-})
    fi
    # without DJL: its engine providers would pull jitnn-djl and DJL into the image
    deps=$(tr ':' '\n' < "$cpfile" | grep -v -e '/ai/djl/' -e 'jitnn-djl' -e '/tornado' | paste -sd: -)
    APP_CP=$root/jitnn-examples/target/classes:$root/jitnn-examples/src/main/resources:$deps
fi
feature=$(ls "$root"/jitnn-native-image/target/jitnn-native-image-*.jar 2>/dev/null | grep -v -e -sources -e -javadoc | head -1 || true)
[ -n "$feature" ] || { echo "no jitnn-native-image jar: ./mvnw install -DskipTests first" >&2; exit 2; }

# TornadoVM's modules, as its argfile adds them
jars=$TORNADOVM_HOME/share/java/tornado
mods=$(sed -n 's/^--add-modules //p' "$TORNADOVM_HOME/tornado-argfile" | tr ',' '\n' | grep '^tornado\.' | paste -sd, -)
[ -n "$mods" ] || { echo "cannot read --add-modules from $TORNADOVM_HOME/tornado-argfile" >&2; exit 2; }
opens=()
for m in tornado.runtime tornado.drivers.cuda tornado.drivers.opencl tornado.drivers.metal; do
    [[ ",$mods," == *",$m,"* ]] && opens+=(--add-opens "java.base/java.lang=$m")
done
jvm=(--enable-native-access="ALL-UNNAMED,$mods" -p "$jars" --add-modules "jdk.unsupported,java.management,jdk.management,$mods" "${opens[@]}")

case $cmd in
build)
    : "${GRAALVM_HOME:?set GRAALVM_HOME to GraalVM for JDK 25 with native-image}"
    out=$here/target/$(basename "${main//.//}")
    if [ "${1:-}" = "-o" ]; then out=$2; shift 2; fi
    # TornadoVM ships no native-image metadata of its own: add the traced copy, and the JDK entries
    # the examples need (ImageIO, Java 2D) when building one of them
    dirs=$here/tornadovm
    [[ $main == ai.beestream.jitnn.examples.* || -n "${NI_EXAMPLES:-}" ]] && dirs=$dirs,$here/examples
    extra=(-H:ConfigurationFileDirectories="$dirs")
    "$GRAALVM_HOME/bin/native-image" "${jvm[@]}" -cp "$APP_CP:$feature" -o "$out" "${extra[@]}" "$@" "$main"
    echo "built $out"
    ;;
trace)
    : "${GRAALVM_HOME:?set GRAALVM_HOME to GraalVM for JDK 25 with native-image}"
    dir=$here/target/trace/$(basename "${main//.//}")
    mkdir -p "$dir"
    "$GRAALVM_HOME/bin/java" -agentlib:native-image-agent=config-merge-dir="$dir" @"$TORNADOVM_HOME/tornado-argfile" \
        ${JAVA_OPTS:-} -cp "$APP_CP" "$main" "$@"
    ;;
java)
    exec "${JAVA_HOME:-$GRAALVM_HOME}/bin/java" @"$TORNADOVM_HOME/tornado-argfile" ${JAVA_OPTS:-} -cp "$APP_CP" "$main" "$@"
    ;;
*)
    sed -n '/^# usage:/,/^set -e/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//' >&2
    exit 2
    ;;
esac
