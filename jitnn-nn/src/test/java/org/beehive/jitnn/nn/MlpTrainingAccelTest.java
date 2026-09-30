package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * The wave 1 gate on a device: the same MLP, same initial weights (counter-based RNG) and same
 * batches trained on TornadoVM and on the host reference give the same loss curve, and each
 * training step after the first replays a cached plan.
 */
class MlpTrainingAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void deviceMatchesReference() {
        for (boolean adam : new boolean[] { true, false }) {
            MlpRun expected = Engine.with(Backend.REFERENCE, () -> MlpRun.train(adam, 4));
            long[] before = Engine.planStats();
            MlpRun actual = Engine.with(Backend.TORNADO, () -> MlpRun.train(adam, 4));
            long[] after = Engine.planStats();

            String name = adam ? "adam" : "sgd";
            assertEquals(expected.losses.size(), actual.losses.size());
            for (int i = 0; i < expected.losses.size(); i++) {
                float e = expected.losses.get(i);
                float a = actual.losses.get(i);
                assertTrue(Math.abs(e - a) <= 1e-3 * Math.max(1, Math.abs(e)), name + " step " + i + ": reference " + e + " device " + a);
            }
            assertEquals(expected.accuracy, actual.accuracy, 0.01, name + " accuracy");
            assertTrue(actual.accuracy > 0.95f, name + " accuracy " + actual.accuracy);

            long compiles = after[0] - before[0];
            long replays = after[1] - before[1];
            int steps = actual.losses.size();
            // 5 parameter inits + the training step + the evaluation pass + reading weights.
            assertTrue(compiles <= 10, name + ": " + compiles + " plans compiled");
            assertTrue(replays >= steps - 1, name + ": only " + replays + " replays for " + steps + " steps");
        }
    }
}
