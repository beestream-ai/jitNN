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

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** TensorFlow models on a TornadoVM device against TensorFlow's outputs. */
class TfGoldenAccelTest {

    @org.junit.jupiter.api.AfterAll
    static void release() {
        Engine.reset();
    }

    @TestFactory
    Stream<DynamicTest> goldens() throws Exception {
        List<DynamicTest> tests = new ArrayList<>();
        for (TfGolden.Case c : TfGolden.load()) {
            tests.add(DynamicTest.dynamicTest(c.name(), () -> Engine.with(Backend.TORNADO, () -> {
                try {
                    TfGolden.check(c, 1e-4, 2e-5);
                } catch (java.io.IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
                return null;
            })));
        }
        return tests.stream();
    }
}
