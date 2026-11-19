#!/usr/bin/env bash
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
# Builds jitNN's API documentation (Javadoc) into the directory given as $1.
#
# Javadoc needs JDK 22+ and the TornadoVM jars jitNN compiles against, which Maven fetches from
# Maven Central (the version in the pom). When no JDK 22+ is there (as on Read the Docs), one is
# downloaded by install-jdk.sh and its checksum verified (Linux x64 only).
#   JAVA_HOME         a JDK 22+ to use; otherwise one is downloaded
#   TORNADOVM_SRC     a TornadoVM checkout to build and document against instead of the release
#   TORNADOVM_VERSION a TornadoVM version already in the local Maven repository (a develop install)
set -euo pipefail

out=${1:?usage: build-javadoc.sh <output directory>}
mkdir -p "$out" && out=$(cd "$out" && pwd)
root=$(cd "$(dirname "$0")/.." && pwd)
work=${JITNN_JAVADOC_WORK:-$root/target/javadoc-build}
mkdir -p "$work"

jdk_feature() { "$1/bin/java" -XshowSettings:properties -version 2>&1 | sed -n 's/^ *java.specification.version = //p'; }

feature=$([ -n "${JAVA_HOME:-}" ] && jdk_feature "$JAVA_HOME" || true)
if ! [[ "$feature" =~ ^[0-9]+$ ]] || [ "$feature" -lt 22 ]; then
    bash "$root/docs/install-jdk.sh" "$work/jdk"
    export JAVA_HOME=$work/jdk
fi
export PATH=$JAVA_HOME/bin:$PATH

if [ -n "${TORNADOVM_SRC:-}" ]; then
    echo "Building TornadoVM's Java modules from $TORNADOVM_SRC"
    (cd "$TORNADOVM_SRC" && ./mvnw -B -q -Pjdk22plus -DskipTests -Dcheckstyle.skip -Dmaven.javadoc.skip install \
        -pl tornado-api,tornado-cublas,tornado-cudnn,tornado-cufft,tornado-cusparse,tornado-cutlass,tornado-mlx,tornado-nccl -am)
    TORNADOVM_VERSION=$(cd "$TORNADOVM_SRC" && ./mvnw -B -q -Pjdk22plus help:evaluate -Dexpression=project.version -DforceStdout)
fi
version=${TORNADOVM_VERSION:+-Dtornadovm.version=$TORNADOVM_VERSION}

echo "Building jitNN's Javadoc against TornadoVM ${TORNADOVM_VERSION:-from the pom}"
cd "$root"
./mvnw -B -q -DskipTests $version install
./mvnw -B -q $version javadoc:aggregate
cp -r target/reports/apidocs/. "$out"
echo "Javadoc in $out"
