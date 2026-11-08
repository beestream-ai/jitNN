package ai.beestream.jitnn.bench;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.Test;

/** Profiles the workloads end to end on the test device; only with {@code -Djitnn.models=true} (see {@link ModelProfile}). */
class ModelProfileAccelTest {

    @Test
    void models() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.models"), "set -Djitnn.models=true");
        assertTrue(ModelProfile.run() > 0, "no workload ran");
    }
}
