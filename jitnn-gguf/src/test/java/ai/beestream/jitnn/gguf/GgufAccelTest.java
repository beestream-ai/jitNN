package ai.beestream.jitnn.gguf;

import java.io.IOException;
import java.util.List;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** GGUF weights, memory-mapped and copied, on the test device. */
class GgufAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @TestFactory
    List<DynamicTest> weightsOnDevice() throws IOException {
        return GgufChecks.weights(Engine.Backend.TORNADO);
    }
}
