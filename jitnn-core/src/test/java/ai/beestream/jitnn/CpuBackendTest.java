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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

import ai.beestream.jitnn.cpu.Cpu;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/** The CPU backend: every golden case, its executor (lazy runs, buffer reuse, counters) and its thread pool. */
class CpuBackendTest {

    @TestFactory
    List<DynamicTest> cpuMatchesPyTorch() {
        return GoldenOps.tests(Backend.CPU);
    }

    @Test
    void operatorsWaitForASync() {
        JitnnRuntime.with(Backend.CPU, () -> {
            Tensor y = Tensor.of(new float[] { -1, 2, -3, 4 }, 4).relu().add(Tensor.of(new float[] { 1, 1, 1, 1 }, 4));
            assertEquals(2, JitnnRuntime.pending(), "relu and add are recorded, not run");
            close(new float[] { 1, 3, 1, 5 }, y.toFloatArray(), 0, 0, "relu + 1");
            assertEquals(0, JitnnRuntime.pending(), "reading the result ran them");
            return null;
        });
    }

    @Test
    void sameAsReferenceThroughTraining() {
        int[] labels = { 1, 0, 2, 1, 2, 0 };
        Supplier<float[]> step = () -> {
            Tensor x = Tensor.uniform(3, -1, 1, 6, 12);
            Tensor w1 = Tensor.uniform(4, -0.3f, 0.3f, 32, 12).requiresGrad(true);
            Tensor b1 = Tensor.uniform(5, -0.1f, 0.1f, 32).requiresGrad(true);
            Tensor w2 = Tensor.uniform(6, -0.3f, 0.3f, 3, 32).requiresGrad(true);
            Tensor loss = F.crossEntropy(F.linear(F.linear(x, w1, b1).relu(), w2, null), Tensor.of(labels, 6));
            loss.backward();
            float[] a = w1.grad().toFloatArray();
            float[] c = w2.grad().toFloatArray();
            float[] all = new float[a.length + c.length + 1];
            System.arraycopy(a, 0, all, 0, a.length);
            System.arraycopy(c, 0, all, a.length, c.length);
            all[all.length - 1] = loss.item();
            return all;
        };
        close(JitnnRuntime.with(Backend.REFERENCE, step::get), JitnnRuntime.with(Backend.CPU, step::get), 1e-5, 1e-6, "MLP loss and gradients");
    }

    @Test
    void inferenceIntermediatesAreReused() {
        Supplier<Tensor> model = () -> {
            Tensor h = Tensor.uniform(7, -1, 1, 64, 256);
            for (int i = 0; i < 6; i++) {
                h = F.linear(h, Tensor.uniform(10 + i, -0.1f, 0.1f, 256, 256), null).relu();
            }
            return h.sum();
        };
        float[] expected = JitnnRuntime.with(Backend.REFERENCE, () -> model.get().toFloatArray());
        Tensor[] middle = new Tensor[1];
        float[] actual = JitnnRuntime.with(Backend.CPU, () -> Grad.inference(() -> {
            Tensor h = Tensor.uniform(7, -1, 1, 64, 256);
            for (int i = 0; i < 6; i++) {
                h = F.linear(h, Tensor.uniform(10 + i, -0.1f, 0.1f, 256, 256), null).relu();
                if (i == 2) {
                    middle[0] = h;
                }
            }
            return h.sum();
        }).toFloatArray());
        close(expected, actual, 1e-5, 1e-5, "six layers");
        assertThrows(IllegalStateException.class, () -> middle[0].toFloatArray(), "an internal intermediate's buffer went back to the pool");
    }

    @Test
    void countsCallsAndReferenceFallbacks() {
        long before = calls("binary");
        JitnnRuntime.with(Backend.CPU, () -> Tensor.of(new float[] { 1, 2 }, 2).add(Tensor.of(new float[] { 3, 4 }, 2)).toFloatArray());
        assertEquals(before + 1, calls("binary"), "one add");
        long[] binary = JitnnRuntime.cpuStats().get("binary");
        assertTrue(binary[2] <= binary[0], "fallbacks are a subset of the calls");
    }

    private static long calls(String op) {
        long[] v = JitnnRuntime.cpuStats().get(op);
        return v == null ? 0 : v[0];
    }

    @Test
    void parallelForCoversEveryIndexOnce() {
        int n = 1_000_003;
        int[] hits = new int[n];
        AtomicInteger chunks = new AtomicInteger();
        Cpu.parallelFor(n, 1000, (from, to) -> {
            chunks.incrementAndGet();
            for (int i = from; i < to; i++) {
                hits[i]++;
            }
        });
        for (int i = 0; i < n; i++) {
            assertEquals(1, hits[i], "index " + i);
        }
        assertTrue(chunks.get() >= 1 && chunks.get() <= Cpu.threads(), "at most one chunk per thread");
        assertTrue(Cpu.threads() <= Runtime.getRuntime().availableProcessors(), "physical cores at most");
    }

    @Test
    void nestedLoopsRunInlineAndFailuresReachTheCaller() {
        AtomicLong sum = new AtomicLong();
        Cpu.parallelFor(64, 1, (from, to) -> {
            for (int i = from; i < to; i++) {
                int row = i;
                Cpu.parallelFor(100, 1, (a, b) -> sum.addAndGet((long) (b - a) * row));
            }
        });
        assertEquals(100L * 63 * 64 / 2, sum.get(), "inner loops ran completely");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Cpu.parallelFor(1000, 1, (from, to) -> {
            if (from <= 999 && 999 < to) {
                throw new IllegalArgumentException("chunk failed");
            }
        }));
        assertEquals("chunk failed", e.getMessage());
        int[] after = new int[10];
        Cpu.parallelFor(10, 1, (from, to) -> {
            for (int i = from; i < to; i++) {
                after[i] = i;
            }
        });
        assertEquals(9, after[9], "the pool still works after a failure");
    }
}
