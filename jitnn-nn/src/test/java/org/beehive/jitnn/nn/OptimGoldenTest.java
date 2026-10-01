package org.beehive.jitnn.nn;

import java.io.IOException;
import java.util.List;

import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

class OptimGoldenTest {

    @TestFactory
    List<DynamicTest> referenceMatchesTorchOptim() throws IOException {
        return OptimGolden.tests(Engine.Backend.REFERENCE);
    }
}
