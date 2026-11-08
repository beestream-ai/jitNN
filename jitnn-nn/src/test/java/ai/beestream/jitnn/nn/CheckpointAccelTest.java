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

import java.nio.file.Path;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Checkpoints on a device: weights and optimiser state are read back from the device to save
 * them and written in place on load, so plans already compiled keep running on the loaded values.
 */
class CheckpointAccelTest {

    @TempDir
    Path dir;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void resumeMatchesEveryOptimizer() {
        for (var make : CheckpointRun.optimizers()) {
            CheckpointRun.resumeMatches(make, dir, 4);
        }
    }

    @Test
    void buffersRoundTrip() {
        CheckpointRun.buffersRoundTrip(dir);
    }
}
