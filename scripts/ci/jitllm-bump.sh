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
# usage: scripts/ci/jitllm-bump.sh [version]
# Moves jitNN to the newest jitllm on Maven Central (io.github.beehive-lab:jitllm), as the
# jitllm-bump workflow (.github/workflows/jitllm-bump.yml) does: it reads jitllm.version from
# jitnn-llm/pom.xml, picks Central's newest release with the same JDK suffix (-jdk22plus), or the
# version given (which must be such a release), and if that is newer replaces the old version in
# every POM and doc that names it.
# Run from the checkout root; it only edits files, never commits.
# Prints, and appends to $GITHUB_OUTPUT when set: current=, latest=, bump=true|false, files=.
# Environment:
#   JITLLM_METADATA   the maven-metadata.xml URL (default: Maven Central's)
set -euo pipefail
pom=jitnn-llm/pom.xml
metadata=${JITLLM_METADATA:-https://repo1.maven.org/maven2/io/github/beehive-lab/jitllm/maven-metadata.xml}

current=$(sed -n 's:.*<jitllm.version>\(.*\)</jitllm.version>.*:\1:p' "$pom")
[ -n "$current" ] || { echo "::error::no <jitllm.version> in $pom"; exit 1; }
suffix=${current#*-}                      # jdk22plus: the TornadoVM build jitNN runs on
[ "$suffix" != "$current" ] || suffix=""

versions=$(curl -fsSL --retry 3 "$metadata" | sed -n 's:.*<version>\(.*\)</version>.*:\1:p') \
    || { echo "::error::could not read $metadata"; exit 1; }
# releases only (X.Y.Z-<suffix>): no -SNAPSHOT, -rc or -dev builds, no other JDK's build
if [ -n "$suffix" ]; then pattern="^[0-9]+\.[0-9]+\.[0-9]+-${suffix}\$"; else pattern='^[0-9]+\.[0-9]+\.[0-9]+$'; fi
releases=$(echo "$versions" | grep -E "$pattern" || true)
if [ -n "${1:-}" ]; then
    latest=$1
    echo "$releases" | grep -qxF -- "$latest" \
        || { echo "::error::$latest is not a jitllm release ending in -$suffix in $metadata"; exit 1; }
else
    latest=$(echo "$releases" | sort -V | tail -1)
    [ -n "$latest" ] || { echo "::error::no jitllm release ending in -$suffix in $metadata"; exit 1; }
fi

newest=$(printf '%s\n%s\n' "$current" "$latest" | sort -V | tail -1)
bump=false
files=""
if [ "$latest" != "$current" ] && [ "$newest" = "$latest" ]; then
    bump=true
    # every POM and doc that names the old version (the property, comments, install snippets)
    files=$(git grep -l -F "$current" -- 'pom.xml' '*/pom.xml' '*.md' 'docs/**' || true)
    for f in $files; do
        # escape the dots so 1.0.2 does not also match 1x0y2
        sed -i.bak "s/$(printf '%s' "$current" | sed 's/[.]/\\./g')/$latest/g" "$f"
        rm -f "$f.bak"
        grep -qF -- "$latest" "$f" || { echo "::error::could not put $latest in $f"; exit 1; }
    done
    files=$(echo $files)
fi

out() { echo "$1=$2"; [ -z "${GITHUB_OUTPUT:-}" ] || echo "$1=$2" >> "$GITHUB_OUTPUT"; }
out current "$current"
out latest "$latest"
out bump "$bump"
out files "$files"
