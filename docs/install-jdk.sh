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
# Installs a Temurin JDK 25 (Linux x64) into the directory given as $1, checking its checksum, so
# that $1 can be used as JAVA_HOME. Read the Docs' build image has no Java; .readthedocs.yaml runs
# this before the build, and build-javadoc.sh runs it when it finds no JDK 22+.
set -euo pipefail

dir=${1:?usage: install-jdk.sh <directory>}
if [ -x "$dir/bin/java" ]; then
    echo "JDK already in $dir"
    exit 0
fi

url=https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.1%2B8/OpenJDK25U-jdk_x64_linux_hotspot_25.0.1_8.tar.gz
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

echo "Downloading JDK 25 into $dir"
mkdir -p "$dir"
tarball=$(mktemp)
trap 'rm -f "$tarball"' EXIT
curl -fsSL -o "$tarball" "$url"
expected=$(curl -fsSL "$url.sha256.txt" | cut -d' ' -f1)
[ "$expected" = "$(sha256 "$tarball")" ] || { echo "JDK download checksum mismatch" >&2; exit 1; }
tar xzf "$tarball" -C "$dir" --strip-components 1
"$dir/bin/java" -version
