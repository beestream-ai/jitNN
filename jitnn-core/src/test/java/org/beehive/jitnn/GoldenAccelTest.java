package org.beehive.jitnn;

import java.util.List;

import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** The device kernels against PyTorch, every golden case. */
class GoldenAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @TestFactory
    List<DynamicTest> deviceMatchesPyTorch() {
        return GoldenOps.tests(Engine.Backend.TORNADO);
    }
}
