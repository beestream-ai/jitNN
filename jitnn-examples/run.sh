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
# Runs one example on the TornadoVM SDK in TORNADOVM_HOME against a released jitNN jar,
# jitnn-<version>.jar: no build needed. The jar is downloaded once into ~/.jitnn/releases and the
# examples are compiled against it once.
#   JITNN_VERSION        the release (default: this checkout's version when it is a release, else
#                        the newest release on GitHub)
#   JITNN_JAR            a jitnn-<version>.jar to use instead of downloading one
#   JITNN_FROM_SOURCE=1  use this checkout's own build instead (./mvnw install -DskipTests first)
#   JAVA_OPTS            JVM flags, such as -Djitnn.device=1:0 or -Djitnn.backend=cpu
#   JITNN_DEVICE_MEMORY  TornadoVM's device-memory budget (default 12GB)
# djl.TrainMnistDjl also needs DJL: its jars are resolved once with the Maven wrapper.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
src=$here/src/main/java
if [ $# -lt 1 ]; then
    sed -n '/^# usage:/,/^set -e/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    echo "examples:"
    ls "$src"/ai/beestream/jitnn/examples/*.java "$src"/ai/beestream/jitnn/examples/djl/*.java \
        | sed -e 's#.*/examples/##' -e 's#\.java$##' -e 's#/#.#' | grep -v '^Examples$' | sed 's/^/  /'
    exit 1
fi
[ -n "$TORNADOVM_HOME" ] || { echo "TORNADOVM_HOME is not set: source TornadoVM's setvars.sh first"; exit 1; }
example=$1
shift
cache=${JITNN_CACHE:-$HOME/.jitnn}

# DJL and its dependencies, for the DJL example only
djl_classpath() {
    local v out
    v=$(sed -n 's#.*<djl.version>\(.*\)</djl.version>.*#\1#p' "$root/pom.xml" | head -1)
    out=$cache/djl-$v.classpath
    if [ ! -s "$out" ]; then
        mkdir -p "$cache/djl-$v"
        cat > "$cache/djl-$v/pom.xml" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>
  <groupId>local</groupId><artifactId>jitnn-examples-djl</artifactId><version>1</version>
  <dependencies>
    <dependency><groupId>ai.djl</groupId><artifactId>basicdataset</artifactId><version>$v</version></dependency>
    <dependency><groupId>ai.djl</groupId><artifactId>model-zoo</artifactId><version>$v</version></dependency>
    <dependency><groupId>org.slf4j</groupId><artifactId>slf4j-simple</artifactId><version>2.0.16</version></dependency>
  </dependencies>
</project>
EOF
        echo "resolving DJL $v" >&2
        (cd "$root" && ./mvnw -q -f "$cache/djl-$v/pom.xml" dependency:build-classpath -Dmdep.outputFile="$out") >&2
    fi
    cat "$out"
}

if [ "${JITNN_FROM_SOURCE:-0}" = 1 ]; then
    cpfile=$here/target/classpath.txt
    if [ ! -f "$cpfile" ] || [ "$here/pom.xml" -nt "$cpfile" ]; then
        (cd "$root" && ./mvnw -q -pl jitnn-examples dependency:build-classpath -Dmdep.outputFile="$cpfile" ${MAVEN_ARGS:-})
    fi
    cp="$here/target/classes:$(cat "$cpfile")"
else
    if [ -n "${JITNN_JAR:-}" ]; then
        jar=$JITNN_JAR
        version=$(basename "$jar" .jar | sed 's/^jitnn-//')
    else
        version=${JITNN_VERSION:-}
        if [ -z "$version" ]; then
            version=$(awk '/<artifactId>jitnn-parent<\/artifactId>/ {p=1; next} p && /<version>/ {gsub(/.*<version>|<\/version>.*/, ""); print; exit}' "$root/pom.xml")
            case $version in *SNAPSHOT*|"")
                version=$(curl -fsSL "https://api.github.com/repos/beestream-ai/jitNN/releases?per_page=1" \
                    | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' | head -1) ;;
            esac
            [ -n "$version" ] || { echo "no jitNN release found: set JITNN_VERSION or JITNN_JAR"; exit 1; }
        fi
        jar=$cache/releases/jitnn-$version.jar
        if [ ! -s "$jar" ]; then
            mkdir -p "$cache/releases"
            echo "downloading jitnn-$version.jar" >&2
            curl -fsSL -o "$jar.part" "https://github.com/beestream-ai/jitNN/releases/download/v$version/jitnn-$version.jar"
            mv "$jar.part" "$jar"
        fi
    fi
    api=$(ls "$TORNADOVM_HOME"/share/java/tornado/tornado-api-*.jar | head -1)
    extra=""
    sources=$(ls "$src"/ai/beestream/jitnn/examples/*.java)
    case $example in djl.*)
        extra=$(djl_classpath)
        sources="$sources $(ls "$src"/ai/beestream/jitnn/examples/djl/*.java)" ;;
    esac
    # compiled again only when the jar or an example changes
    classes=$cache/examples-classes/$version${extra:+-djl}
    stamp=$classes/.compiled
    if [ ! -f "$stamp" ] || [ -n "$(find "$src" "$jar" -newer "$stamp" -print -quit)" ]; then
        rm -rf "$classes" && mkdir -p "$classes"
        echo "compiling the examples against jitnn-$version.jar" >&2
        javac -nowarn -d "$classes" -cp "$jar:$api${extra:+:$extra}" $sources
        touch "$stamp"
    fi
    cp="$classes:$here/src/main/resources:$jar${extra:+:$extra}"
fi
exec java @"$TORNADOVM_HOME"/tornado-argfile --add-modules jdk.incubator.vector \
    -Dtornado.device.memory=${JITNN_DEVICE_MEMORY:-12GB} ${JAVA_OPTS:-} \
    -cp "$cp" ai.beestream.jitnn.examples."$example" "$@"
