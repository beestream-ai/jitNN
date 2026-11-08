#!/usr/bin/env bash
# Builds jitNN's API documentation (Javadoc) into the directory given as $1.
#
# Javadoc needs JDK 22+ and the TornadoVM jars jitNN compiles against. When they are not there (as on
# Read the Docs), the script downloads a JDK and builds TornadoVM's Java modules from source:
#   JAVA_HOME         a JDK 22+ to use; otherwise one is downloaded
#   TORNADOVM_REF     the TornadoVM branch or tag to build (default: develop)
#   TORNADOVM_SRC     an existing TornadoVM checkout to build instead of cloning one
#   TORNADOVM_VERSION skip building TornadoVM: this version is already in the local Maven repository
set -euo pipefail

out=${1:?usage: build-javadoc.sh <output directory>}
mkdir -p "$out" && out=$(cd "$out" && pwd)
root=$(cd "$(dirname "$0")/.." && pwd)
work=${JITNN_JAVADOC_WORK:-$root/target/javadoc-build}
mkdir -p "$work"

jdk_feature() { "$1/bin/java" -XshowSettings:properties -version 2>&1 | sed -n 's/^ *java.specification.version = //p'; }

feature=$([ -n "${JAVA_HOME:-}" ] && jdk_feature "$JAVA_HOME" || true)
if ! [[ "$feature" =~ ^[0-9]+$ ]] || [ "$feature" -lt 22 ]; then
    if [ ! -x "$work/jdk/bin/java" ]; then
        echo "Downloading JDK 25"
        url=https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.1%2B8/OpenJDK25U-jdk_x64_linux_hotspot_25.0.1_8.tar.gz
        mkdir -p "$work/jdk"
        curl -fsSL "$url" | tar xz -C "$work/jdk" --strip-components 1
    fi
    export JAVA_HOME=$work/jdk
fi
export PATH=$JAVA_HOME/bin:$PATH

if [ -z "${TORNADOVM_VERSION:-}" ]; then
    src=${TORNADOVM_SRC:-$work/TornadoVM}
    if [ ! -d "$src/.git" ] && [ -z "${TORNADOVM_SRC:-}" ]; then
        echo "Cloning TornadoVM ${TORNADOVM_REF:-develop}"
        git clone -q --depth 1 -b "${TORNADOVM_REF:-develop}" https://github.com/beehive-lab/TornadoVM "$src"
    fi
    echo "Building TornadoVM's Java modules"
    (cd "$src" && ./mvnw -B -q -Pjdk22plus -DskipTests -Dcheckstyle.skip -Dmaven.javadoc.skip install \
        -pl tornado-api,tornado-cublas,tornado-cudnn,tornado-cufft,tornado-cusparse,tornado-cutlass,tornado-mlx,tornado-nccl -am)
    TORNADOVM_VERSION=$(cd "$src" && ./mvnw -B -q -Pjdk22plus help:evaluate -Dexpression=project.version -DforceStdout)
fi

echo "Building jitNN's Javadoc against TornadoVM $TORNADOVM_VERSION"
cd "$root"
./mvnw -B -q -DskipTests -Dtornadovm.version="$TORNADOVM_VERSION" install
./mvnw -B -q -Dtornadovm.version="$TORNADOVM_VERSION" javadoc:aggregate
cp -r target/reports/apidocs/. "$out"
echo "Javadoc in $out"
