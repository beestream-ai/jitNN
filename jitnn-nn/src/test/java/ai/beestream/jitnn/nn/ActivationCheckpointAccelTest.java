package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Activation checkpointing on a device: the same loss and gradients as without it, and less
 * device memory for intermediates in the training step's plan.
 */
class ActivationCheckpointAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void sameGradients() {
        for (ActivationCheckpointRun.Case c : ActivationCheckpointRun.cases(2, 16, 32, 2)) {
            ActivationCheckpointRun.same(c, () -> ActivationCheckpointRun.step(c.model().apply(false), c.loss()),
                    () -> ActivationCheckpointRun.step(c.model().apply(true), c.loss()), 1e-5);
            Engine.reset();
        }
    }

    @Test
    void lessMemory() {
        int b = Integer.getInteger("jitnn.bench.batch", 8);
        int t = Integer.getInteger("jitnn.bench.seq", 256);
        int dim = Integer.getInteger("jitnn.bench.dim", 512);
        int layers = Integer.getInteger("jitnn.bench.layers", 8);
        ActivationCheckpointRun.Case gpt2 = ActivationCheckpointRun.cases(b, t, dim, layers).get(0);
        long[] bytes = new long[2];
        double[] ms = new double[2];
        for (int ck = 0; ck < 2; ck++) {
            Module m = gpt2.model().apply(ck == 1);
            for (int i = 0; i < 3; i++) {
                ActivationCheckpointRun.step(m, gpt2.loss());
                m.zeroGrad();
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < 5; i++) {
                ActivationCheckpointRun.step(m, gpt2.loss());
                m.zeroGrad();
            }
            ms[ck] = (System.nanoTime() - t0) / 5e6;
            bytes[ck] = Engine.lastIntermediateBytes();
            Engine.reset();
        }
        System.out.printf("[checkpoint] GPT-2 (dim %d, %d layers, %d x %d tokens): intermediates %.1f MB -> %.1f MB, step %.1f ms -> %.1f ms%n", dim, layers, b, t,
                bytes[0] / 1e6, bytes[1] / 1e6, ms[0], ms[1]);
        assertTrue(bytes[1] < 0.6 * bytes[0], "checkpointing saved too little: " + bytes[0] + " -> " + bytes[1]);
    }
}
