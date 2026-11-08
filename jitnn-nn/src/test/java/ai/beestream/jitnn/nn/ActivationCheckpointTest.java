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
package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Activation checkpointing on the reference backend: the same loss and gradients, bit for bit. */
class ActivationCheckpointTest {

    @BeforeAll
    static void reference() {
        Engine.setBackend(Engine.Backend.REFERENCE);
    }

    @Test
    void sameGradients() {
        for (ActivationCheckpointRun.Case c : ActivationCheckpointRun.cases(2, 8, 16, 2)) {
            ActivationCheckpointRun.same(c, () -> ActivationCheckpointRun.step(c.model().apply(false), c.loss()),
                    () -> ActivationCheckpointRun.step(c.model().apply(true), c.loss()), 0.0);
        }
    }
}
