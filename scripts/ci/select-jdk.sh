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
# usage: scripts/ci/select-jdk.sh
# Sets JAVA_HOME and PATH for the later steps of a GitHub Actions job: JAVA_OVERRIDE (the
# JITNN_JAVA_HOME repository variable) if set, else SDKMAN!'s current java.
set -euo pipefail
java_home=${JAVA_OVERRIDE:-${SDKMAN_DIR:-$HOME/.sdkman}/candidates/java/current}
[ -x "$java_home/bin/java" ] || { echo "::error::no JDK at $java_home (set JITNN_JAVA_HOME)"; exit 1; }
echo "JAVA_HOME=$java_home" >> "$GITHUB_ENV"
echo "$java_home/bin" >> "$GITHUB_PATH"
"$java_home/bin/java" -version
