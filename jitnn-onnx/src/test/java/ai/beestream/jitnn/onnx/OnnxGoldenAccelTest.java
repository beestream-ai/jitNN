package ai.beestream.jitnn.onnx;

import java.io.IOException;
import java.util.List;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** The ONNX goldens on the test device. */
class OnnxGoldenAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @TestFactory
    List<DynamicTest> deviceMatchesOnnxRuntime() throws IOException {
        return OnnxGolden.tests(Engine.Backend.TORNADO);
    }
}
