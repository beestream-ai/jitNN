package org.beehive.jitnn.nn;

import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Activation checkpointing on the reference backend: the same loss and gradients, bit for bit. */
class ActivationCheckpointTest {

    @BeforeAll
    static void reference() {
        Engine.setBackend(Engine.Backend.REFERENCE);
    }

    @Test
    void sameGradients() {
        for (ActivationCheckpointRun.Case c : ActivationCheckpointRun.cases(2, 8, 16, 2)) {
            ActivationCheckpointRun.same(c, () -> ActivationCheckpointRun.step(c.model().apply(false), c.loss()),
                    () -> ActivationCheckpointRun.step(c.model().apply(true), c.loss()), 0.0);
        }
    }
}
