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
package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;

/**
 * Closing a module releases its weights and the folded copies jitNN made of them; a second model
 * of the same structure reuses the closed model's plan without compiling.
 */
class ModuleCloseAccelTest {

    @AfterEach
    void release() {
        JitnnRuntime.reset();
    }

    private static Sequential model(long seed) {
        Sequential m = new Sequential(new Conv2d(3, 16, 3, 1, 1, seed), new BatchNorm2d(16), new ReLU(), new Conv2d(16, 8, 3, 1, 1, seed + 1));
        m.train(false);
        return m;
    }

    private static float[] run(Module m, Tensor x) {
        return Grad.inference(() -> m.forward(x)).toFloatArray();
    }

    @Test
    void closedModelsReleaseTheirWeightsAndLeaveTheirPlansForTheNext() {
        JitnnRuntime.with(Backend.TORNADO, () -> {
            float[] input = new float[2 * 3 * 16 * 16];
            for (int i = 0; i < input.length; i++) {
                input[i] = (i % 17) * 0.05f - 0.4f;
            }
            Tensor x = Tensor.of(input, 2, 3, 16, 16);
            Sequential a = model(1);
            float[] ya = run(a, x);
            run(a, x);
            a.close();
            assertTrue(a.isClosed());
            assertThrows(IllegalStateException.class, () -> run(a, x));
            a.close(); // twice: nothing

            Sequential b = model(1); // the same weights, built again
            float[] expected = JitnnRuntime.with(Backend.REFERENCE, () -> run(b, x));
            long[] before = JitnnRuntime.planStats();
            float[] yb = run(b, x);
            long[] after = JitnnRuntime.planStats();
            assertArrayEquals(ya, yb, 1e-5f);
            assertArrayEquals(expected, yb, 1e-3f);
            assertEquals(before[0], after[0], "the second model compiled a plan instead of reusing the closed model's");
            b.close();
            return null;
        });
    }
}
