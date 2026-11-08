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
package ai.beestream.jitnn.engine;

import ai.beestream.jitnn.Tensor;

/** Runs one node on the host with the operator's Java reference implementation. */
final class ReferenceExecutor {

    private ReferenceExecutor() {
    }

    static void run(Node node) {
        for (Tensor t : node.inputs) {
            Engine.ensureHost(t.storage());
        }
        for (Tensor t : node.outputs) {
            t.storage().allocate();
            t.storage().hostWritten();
        }
        node.op.reference(node);
        node.executed = true;
    }
}
