package org.beehive.jitnn.tf;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
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
