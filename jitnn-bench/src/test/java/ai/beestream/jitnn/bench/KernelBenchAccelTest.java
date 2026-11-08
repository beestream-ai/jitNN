package ai.beestream.jitnn.bench;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.Test;

/** Runs the kernel benchmark on the test device; only with {@code -Djitnn.kbench=true} (see {@link KernelBench}). */
class KernelBenchAccelTest {

    @Test
    void kernels() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.kbench"), "set -Djitnn.kbench=true");
        assertTrue(KernelBench.run() > 0, "no call was timed");
    }
}
