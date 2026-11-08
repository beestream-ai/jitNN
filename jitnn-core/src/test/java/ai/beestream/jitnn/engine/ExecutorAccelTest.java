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
package ai.beestream.jitnn.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine.Backend;

/**
 * The TornadoVM executor's parts, through their effects: the plan cache stays within its bound,
 * a closed failed plan leaves its unread outputs readable and its trace compiles again, and a
 * remembered trace sees weights written from the host.
 */
class ExecutorAccelTest {

    @AfterEach
    void release() {
        Engine.reset();
    }

    private static float[] ramp(int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.5f - 3;
        }
        return a;
    }

    private static int localPlans() {
        Storage.LOCK.lock();
        try {
            return (int) PlanCache.CACHE.values().stream().filter(PlanCache::local).count();
        } finally {
            Storage.LOCK.unlock();
        }
    }

    @Test
    void planCacheStaysWithinItsBound() {
        int limit = Math.max(TornadoExecutor.CACHE_SIZE, PlanCache.RESERVED.get()) * TornadoExecutor.SLOTS;
        Engine.with(Backend.TORNADO, () -> {
            for (int k = 1; k <= 3 * TornadoExecutor.CACHE_SIZE; k++) {
                int n = 64 * k; // a new size every time: a new plan
                float[] x = ramp(n);
                float[] y = Tensor.of(x, n).mul(2).add(1).toFloatArray();
                for (int i = 0; i < n; i++) {
                    assertEquals(2 * x[i] + 1, y[i], 1e-6f);
                }
                assertTrue(localPlans() <= limit, "plans " + localPlans() + " over the bound " + limit);
            }
            return null;
        });
    }

    @Test
    void closingAFailedPlanKeepsItsOutputsAndTheTraceCompilesAgain() {
        Engine.with(Backend.TORNADO, () -> {
            int n = 1000;
            float[] x = ramp(n);
            Tensor y = Tensor.of(x, n).mul(3);
            float sum = y.sum().item(); // runs the plan; y itself is not read yet
            CompiledTrace ct;
            Storage.LOCK.lock();
            try {
                ct = PlanCache.CACHE.values().stream().filter(c -> c.bound != null && java.util.Arrays.asList(c.bound).contains(y.storage())).findFirst().orElseThrow();
                PlanCache.CACHE.remove(ct.key);
                PlanRecovery.closeFailed(ct);
            } finally {
                Storage.LOCK.unlock();
            }
            assertTrue(ct.closed);
            assertTrue(!y.storage().validOn.contains(ct));
            float[] expect = new float[n];
            double s = 0;
            for (int i = 0; i < n; i++) {
                expect[i] = 3 * x[i];
                s += expect[i];
            }
            assertArrayEquals(expect, y.toFloatArray(), 1e-6f); // saved to the host when the plan closed
            assertEquals(s, sum, 1e-3);
            long compiles = Telemetry.compiles;
            float again = Tensor.of(x, n).mul(3).sum().item();
            assertEquals(sum, again, 1e-3);
            assertTrue(Telemetry.compiles > compiles, "the closed plan's trace compiled again");
            return null;
        });
    }

    @Test
    void rememberedTraceSeesHostWrittenWeights() {
        Engine.with(Backend.TORNADO, () -> {
            int n = 256;
            Tensor w = Tensor.persistent(Tensor.of(ramp(n), n));
            float[] x = new float[n];
            java.util.Arrays.fill(x, 1);
            float first = 0;
            for (int i = 0; i < 4; i++) { // the third and later runs replay the remembered plan
                first = Tensor.of(x, n).mul(w).sum().item();
            }
            float[] doubled = ramp(n);
            for (int i = 0; i < n; i++) {
                doubled[i] *= 2;
            }
            w.copyFrom(doubled);
            float second = Tensor.of(x, n).mul(w).sum().item();
            assertEquals(2 * first, second, 1e-2f);
            w.release();
            return null;
        });
    }
}
