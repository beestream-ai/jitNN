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

import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.JitnnRuntime;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** The MLP trains on the host reference backend. */
class MlpTrainingTest {

    @BeforeAll
    static void reference() {
        JitnnRuntime.setBackend(Backend.REFERENCE);
    }

    @Test
    void adamConverges() {
        MlpRun run = MlpRun.train(true, 4);
        assertTrue(run.losses.get(run.losses.size() - 1) < 0.3f * run.losses.get(0), "loss " + run.losses);
        assertTrue(run.accuracy > 0.95f, "accuracy " + run.accuracy);
    }

    @Test
    void sgdMomentumConverges() {
        MlpRun run = MlpRun.train(false, 4);
        assertTrue(run.losses.get(run.losses.size() - 1) < 0.3f * run.losses.get(0), "loss " + run.losses);
        assertTrue(run.accuracy > 0.95f, "accuracy " + run.accuracy);
    }
}
