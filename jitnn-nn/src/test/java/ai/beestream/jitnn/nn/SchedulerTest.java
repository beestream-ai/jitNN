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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.optim.LRScheduler;
import ai.beestream.jitnn.nn.optim.SGD;
import org.junit.jupiter.api.Test;

/** Learning-rate schedules against the closed forms of PyTorch and Hugging Face. */
class SchedulerTest {

    private static SGD optimizer() {
        return new SGD(List.of(Tensor.parameter(Tensor.zeros(1))), 0.1f);
    }

    @Test
    void warmupLinear() {
        LRScheduler s = LRScheduler.warmupLinear(optimizer(), 10, 100);
        assertEquals(0.0f, s.currentRate(), 1e-9f);
        float[] expected = { 0.05f, 0.1f, 0.05f, 0.0f, 0.0f };
        int[] steps = { 5, 10, 55, 100, 120 };
        int done = 0;
        for (int i = 0; i < steps.length; i++) {
            for (; done < steps[i]; done++) {
                s.step();
            }
            assertEquals(expected[i], s.currentRate(), 1e-7f, "step " + steps[i]);
        }
    }

    @Test
    void warmupCosine() {
        LRScheduler s = LRScheduler.warmupCosine(optimizer(), 10, 110, 0.1f);
        assertEquals(0.0f, s.currentRate(), 1e-9f);
        int[] steps = { 5, 10, 60, 110, 200 };
        float[] expected = { 0.05f, 0.1f, 0.1f * (0.1f + 0.9f * 0.5f), 0.01f, 0.01f };
        int done = 0;
        for (int i = 0; i < steps.length; i++) {
            for (; done < steps[i]; done++) {
                s.step();
            }
            assertEquals(expected[i], s.currentRate(), 1e-7f, "step " + steps[i]);
        }
    }
}
