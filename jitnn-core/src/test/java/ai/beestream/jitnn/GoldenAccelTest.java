package ai.beestream.jitnn;

import java.util.List;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** The device kernels (or the CPU backend, {@code -Djitnn.test.backend=cpu}) against PyTorch, every golden case. */
class GoldenAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @TestFactory
    List<DynamicTest> deviceMatchesPyTorch() {
        return GoldenOps.tests(OpsAccelTest.TARGET);
    }
}
