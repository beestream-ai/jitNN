package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** The wave 1 MLP trains on the host reference backend. */
class MlpTrainingTest {

    @BeforeAll
    static void reference() {
        Engine.setBackend(Engine.Backend.REFERENCE);
    }

    @Test
    void adamConverges() {
        MlpRun run = MlpRun.train(true, 4);
        assertTrue(run.losses.get(run.losses.size() - 1) < 0.3f * run.losses.get(0), "loss " + run.losses);
        assertTrue(run.accuracy > 0.95f, "accuracy " + run.accuracy);
    }

    @Test
    void sgdMomentumConverges() {
        MlpRun run = MlpRun.train(false, 4);
        assertTrue(run.losses.get(run.losses.size() - 1) < 0.3f * run.losses.get(0), "loss " + run.losses);
        assertTrue(run.accuracy > 0.95f, "accuracy " + run.accuracy);
    }
}
