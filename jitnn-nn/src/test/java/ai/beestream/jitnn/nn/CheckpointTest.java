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

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.JitnnRuntime;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Checkpoints on the host reference backend: resuming training reproduces an uninterrupted run. */
class CheckpointTest {

    @TempDir
    Path dir;

    @BeforeAll
    static void reference() {
        JitnnRuntime.setBackend(Backend.REFERENCE);
    }

    @Test
    void resumeMatchesEveryOptimizer() throws Exception {
        for (var make : CheckpointRun.optimizers()) {
            CheckpointRun.resumeMatches(make, dir, 4);
        }
    }

    @Test
    void buffersRoundTrip() throws Exception {
        CheckpointRun.buffersRoundTrip(dir);
    }

    @Test
    void mismatchesAreRejected() throws Exception {
        CheckpointRun.mismatchesAreRejected(dir);
    }
}
