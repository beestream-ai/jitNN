/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.tf;

/** Small helpers on TensorFlow tensor references. */
final class TfGraphs {

    private TfGraphs() {
    }

    /** "node" as "node:0"; "node:k" unchanged; control inputs keep their "^". */
    static String canonical(String ref) {
        if (ref.startsWith("^") || ref.lastIndexOf(':') > 0) {
            return ref;
        }
        return ref + ":0";
    }

    /** The node a reference reads. */
    static String node(String ref) {
        int c = ref.lastIndexOf(':');
        return c > 0 ? ref.substring(0, c) : ref;
    }
}
