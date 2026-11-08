package ai.beestream.jitnn.djl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.function.Supplier;

import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/** DJL training programs on a TornadoVM device, step by step against the host reference. */
class DjlEngineAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static void same(String what, Supplier<List<Float>> program, double tol) {
        List<Float> expected = Engine.with(Backend.REFERENCE, program);
        List<Float> actual = Engine.with(Backend.TORNADO, program);
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            double e = expected.get(i);
            double a = actual.get(i);
            assertEquals(e, a, tol * Math.max(1, Math.abs(e)), what + " loss at step " + i + ": " + expected + " vs " + actual);
        }
        System.out.println("[djl] " + what + " losses " + actual.get(0) + " -> " + actual.get(actual.size() - 1));
    }

    @Test
    void mlpAdam() {
        same("MLP Adam", () -> DjlWorkloads.trainMlp(30, Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.02f)).build()), 1e-3);
    }

    @Test
    void mlpSgd() {
        same("MLP SGD", () -> DjlWorkloads.trainMlp(30, Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.1f)).build()), 1e-3);
    }

    @Test
    void cnnBatchNorm() {
        same("CNN BatchNorm", () -> DjlWorkloads.trainCnn(20), 2e-3);
    }

    @Test
    void lstm() {
        same("LSTM", () -> DjlWorkloads.trainLstm(20), 2e-3);
    }
}
