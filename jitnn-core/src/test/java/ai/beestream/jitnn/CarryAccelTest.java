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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Values a plan hands to its own next execution ({@code jitnn.plan.carry}): a running sum and
 * gradients accumulated over steps stay on the device once the plan carries them, with the values
 * of the host path, and a value from the host in between still reaches the plan. Run with
 * {@code -Paccel-tests}.
 */
class CarryAccelTest {

    private static final int N = 4096;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static float[] input(int call) {
        float[] v = new float[N];
        for (int i = 0; i < N; i++) {
            v[i] = call + (i % 97) * 0.25f;
        }
        return v;
    }

    @Test
    void runningSumStaysOnTheDevice() {
        Engine.with(Backend.TORNADO, () -> {
            float[] expected = new float[N];
            Tensor acc = Tensor.of(new float[N], 64, 64);
            long fetches = 0;
            for (int c = 0; c < 10; c++) {
                acc = acc.add(Tensor.of(input(c), 64, 64));
                Engine.sync();
                float[] x = input(c);
                for (int i = 0; i < N; i++) {
                    expected[i] += x[i];
                }
                if (c == 4) {
                    // by now the plan carries the sum: later steps fetch nothing to the host
                    fetches = Engine.transferStats()[0];
                }
            }
            assertEquals(fetches, Engine.transferStats()[0], "fetches to the host once the sum is carried");
            assertArrayEquals(expected, acc.toFloatArray(), 1e-3f);
            return null;
        });
    }

    @Test
    void aHostValueInBetweenReachesThePlan() {
        Engine.with(Backend.TORNADO, () -> {
            float[] expected = new float[N];
            Tensor acc = Tensor.of(new float[N], 64, 64);
            for (int c = 0; c < 10; c++) {
                if (c == 6) {
                    // the same plan, but its input comes from the host this time
                    acc = Tensor.of(input(100), 64, 64);
                    expected = input(100);
                }
                acc = acc.add(Tensor.of(input(c), 64, 64));
                Engine.sync();
                float[] x = input(c);
                for (int i = 0; i < N; i++) {
                    expected[i] += x[i];
                }
            }
            assertArrayEquals(expected, acc.toFloatArray(), 1e-3f);
            return null;
        });
    }

    @Test
    void gradientsAccumulatedOverSteps() {
        Engine.with(Backend.TORNADO, () -> {
            Tensor w = Tensor.parameter(Tensor.of(input(1), 64, 64));
            float[] x = input(2);
            int steps = 8;
            for (int s = 0; s < steps; s++) {
                // d/dw sum(w * x) = x, added to the gradient of the earlier steps
                w.mul(Tensor.of(x, 64, 64)).sum().backward();
                Engine.sync();
            }
            float[] expected = new float[N];
            for (int i = 0; i < N; i++) {
                expected[i] = steps * x[i];
            }
            assertArrayEquals(expected, w.grad().toFloatArray(), 1e-3f);
            return null;
        });
    }
}
