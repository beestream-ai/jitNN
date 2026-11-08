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
package ai.beestream.jitnn.gguf;

import java.io.IOException;
import java.util.List;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.JitnnRuntime;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** GGUF weights, memory-mapped and copied, on the test device. */
class GgufAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    @TestFactory
    List<DynamicTest> weightsOnDevice() throws IOException {
        return GgufChecks.weights(Backend.TORNADO);
    }
}
