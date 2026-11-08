package ai.beestream.jitnn.nn;

import java.io.IOException;
import java.util.List;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

class OptimGoldenAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @TestFactory
    List<DynamicTest> deviceMatchesTorchOptim() throws IOException {
        return OptimGolden.tests(Engine.Backend.TORNADO);
    }
}
