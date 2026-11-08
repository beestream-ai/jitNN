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
package ai.beestream.jitnn;

import static ai.beestream.jitnn.Checks.close;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Several threads running inference on the device at the same time, each with its own inputs and
 * the shared weights, against results computed one at a time. Run with
 * {@code -Djitnn.test.extra=-Djitnn.concurrency=4} for plans that run in parallel; with the default
 * of 1 the threads take turns.
 */
class ConcurrencyAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static final Tensor W1 = Tensor.persistent(Tensor.uniform(1, -0.05f, 0.05f, 512, 256));
    private static final Tensor B1 = Tensor.persistent(Tensor.uniform(2, -0.1f, 0.1f, 512));
    private static final Tensor W2 = Tensor.persistent(Tensor.uniform(3, -0.05f, 0.05f, 64, 512));
    private static final Tensor B2 = Tensor.persistent(Tensor.uniform(4, -0.1f, 0.1f, 64));

    private static float[] forward(int seed) {
        return Grad.inference(() -> {
            Tensor x = Tensor.of(Checks.seq(128 * 256, 0.01f * seed, 0.001f * seed), 128, 256);
            Tensor h = F.linear(x, W1, B1).gelu();
            return F.layerNorm(F.linear(h, W2, B2), Tensor.ones(64), Tensor.zeros(64), 1e-5f).softmax();
        }).toFloatArray();
    }

    @Test
    void threadsRunInferenceTogether() throws Exception {
        int threads = 4;
        int calls = 25;
        float[][] expected = new float[threads * calls][];
        Engine.with(Backend.TORNADO, () -> {
            for (int i = 0; i < expected.length; i++) {
                expected[i] = forward(i + 1);
            }
            return null;
        });
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<float[][]>> results = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                int first = t * calls;
                results.add(pool.submit(() -> Engine.with(Backend.TORNADO, () -> {
                    float[][] out = new float[calls][];
                    for (int c = 0; c < calls; c++) {
                        out[c] = forward(first + c + 1);
                    }
                    return out;
                })));
            }
            for (int t = 0; t < threads; t++) {
                float[][] out = results.get(t).get();
                for (int c = 0; c < calls; c++) {
                    close(expected[t * calls + c], out[c], 1e-5, 1e-6, "thread " + t + " call " + c);
                }
            }
        } finally {
            pool.shutdown();
        }
    }
}
