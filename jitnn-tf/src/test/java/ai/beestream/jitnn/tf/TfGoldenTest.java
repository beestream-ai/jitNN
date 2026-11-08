package ai.beestream.jitnn.tf;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** TensorFlow models on jitNN's host reference against TensorFlow's outputs. */
class TfGoldenTest {

    @TestFactory
    Stream<DynamicTest> goldens() throws Exception {
        List<DynamicTest> tests = new ArrayList<>();
        for (TfGolden.Case c : TfGolden.load()) {
            tests.add(DynamicTest.dynamicTest(c.name(), () -> Engine.with(Backend.REFERENCE, () -> {
                try {
                    TfGolden.check(c, 1e-4, 1e-5);
                } catch (java.io.IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
                return null;
            })));
        }
        return tests.stream();
    }
}
