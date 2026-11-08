/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
