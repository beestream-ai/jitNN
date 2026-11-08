package ai.beestream.jitnn.nn;

import java.io.IOException;
import java.util.List;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

class OptimGoldenTest {

    @TestFactory
    List<DynamicTest> referenceMatchesTorchOptim() throws IOException {
        return OptimGolden.tests(Engine.Backend.REFERENCE);
    }
}
