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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.Json;
import ai.beestream.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The timeline of a few training steps: valid Chrome trace JSON with one range per step, one host
 * span per plan execution and the kernels on the device thread, whose durations add up to the
 * profiler's device time. Needs {@code -Djitnn.profile=true -Djitnn.cudaGraph=false}.
 */
class TraceAccelTest {

    @TempDir
    Path dir;

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    @Test
    void trainingTimeline() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.profile"), "set -Djitnn.profile=true -Djitnn.cudaGraph=false");
        Sequential model = new Sequential(new Linear(64, 256, 1), new ReLU(), new Linear(256, 10, 2));
        Adam opt = new Adam(model.parameters(), 1e-3f);
        Tensor x = Tensor.uniform(3, -1, 1, 128, 64);
        int[] labels = new int[128];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i % 10;
        }
        Tensor y = Tensor.of(labels, 128);
        JitnnRuntime.startTrace();
        long deviceNs = 0;
        for (int step = 0; step < 4; step++) {
            try (AutoCloseable r = JitnnRuntime.traceRange("step " + step)) {
                Tensor loss = F.crossEntropy(model.forward(x), y);
                loss.backward();
                opt.step();
                loss.item();
                opt.zeroGrad();
            }
            deviceNs += JitnnRuntime.lastProfile()[0];
        }
        Path file = dir.resolve("trace.json");
        JitnnRuntime.stopTrace(file);

        Map<?, ?> trace = (Map<?, ?>) Json.parse(file);
        List<?> events = (List<?>) trace.get("traceEvents");
        int ranges = 0;
        int plans = 0;
        int kernels = 0;
        double kernelUs = 0;
        for (Object o : events) {
            Map<?, ?> e = (Map<?, ?>) o;
            if (!"X".equals(e.get("ph"))) {
                continue;
            }
            long tid = ((Number) e.get("tid")).longValue();
            double dur = ((Number) e.get("dur")).doubleValue();
            assertTrue(dur >= 0, "negative duration");
            if (tid == 0) {
                ranges++;
            } else if (tid == 1) {
                plans++;
            } else {
                kernels++;
                kernelUs += dur;
            }
        }
        System.out.printf("[trace] %d events: %d ranges, %d plan executions, %d kernels (%.3f ms of device time; profiler %.3f ms)%n", events.size(), ranges, plans, kernels,
                kernelUs / 1e3, deviceNs / 1e6);
        assertEquals(4, ranges);
        assertTrue(plans >= 4, "plan executions " + plans);
        assertTrue(kernels >= 4 * 5, "kernels " + kernels);
        assertEquals(deviceNs / 1e3, kernelUs, 0.02 * deviceNs / 1e3 + 1, "device time in the timeline vs the profiler");
    }
}
