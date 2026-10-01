package org.beehive.jitnn;

import java.util.List;

import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/** The host reference against PyTorch, every golden case. */
class GoldenTest {

    @TestFactory
    List<DynamicTest> referenceMatchesPyTorch() {
        return GoldenOps.tests(Engine.Backend.REFERENCE);
    }
}
