package ai.beestream.jitnn.onnx;

import java.io.IOException;
import java.util.List;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** The ONNX goldens on jitNN's host reference backend. */
class OnnxGoldenTest {

    @TestFactory
    List<DynamicTest> referenceMatchesOnnxRuntime() throws IOException {
        return OnnxGolden.tests(Engine.Backend.REFERENCE);
    }
}
