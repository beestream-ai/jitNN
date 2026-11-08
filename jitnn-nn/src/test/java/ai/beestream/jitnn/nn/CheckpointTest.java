package ai.beestream.jitnn.nn;

import java.nio.file.Path;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Checkpoints on the host reference backend: resuming training reproduces an uninterrupted run. */
class CheckpointTest {

    @TempDir
    Path dir;

    @BeforeAll
    static void reference() {
        Engine.setBackend(Engine.Backend.REFERENCE);
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

    @Test
    void mismatchesAreRejected() {
        CheckpointRun.mismatchesAreRejected(dir);
    }
}
