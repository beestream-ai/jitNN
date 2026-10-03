package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.optim.LRScheduler;
import org.beehive.jitnn.nn.optim.SGD;
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
