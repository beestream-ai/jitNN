package ai.beestream.jitnn.nn;

import java.nio.file.Path;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Checkpoints on a device: weights and optimiser state are read back from the device to save
 * them and written in place on load, so plans already compiled keep running on the loaded values.
 */
class CheckpointAccelTest {

    @TempDir
    Path dir;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void resumeMatchesEveryOptimizer() {
        for (var make : CheckpointRun.optimizers()) {
            CheckpointRun.resumeMatches(make, dir, 4);
        }
    }

    @Test
    void buffersRoundTrip() {
        CheckpointRun.buffersRoundTrip(dir);
    }
}
