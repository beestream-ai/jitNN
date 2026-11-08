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

import static ai.beestream.jitnn.CustomOpFixtures.HYPOT;
import static ai.beestream.jitnn.CustomOpFixtures.MIN_MAX;
import static ai.beestream.jitnn.CustomOpFixtures.REPEAT_SCALED;
import static ai.beestream.jitnn.CustomOpFixtures.SWISH;
import static ai.beestream.jitnn.OpsAccelTest.same;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/** Custom operators on the TornadoVM device against their host references, and kernel printing. */
class CustomOpAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    @Test
    void values() {
        same("custom swish", () -> SWISH.apply(Tensor.uniform(1, -4, 4, 1000, 37)));
        same("custom hypot", () -> HYPOT.apply(Tensor.uniform(2, -4, 4, 3001), Tensor.uniform(3, -4, 4, 3001)));
        same("custom min", () -> MIN_MAX.applyAll(null, null, Tensor.uniform(4, -1, 1, 700, 33))[0]);
        same("custom max", () -> MIN_MAX.applyAll(null, null, Tensor.uniform(4, -1, 1, 700, 33))[1]);
        same("custom repeat", () -> REPEAT_SCALED.apply(new int[] { 5 }, new float[] { -0.5f }, Tensor.uniform(5, -1, 1, 300)));
        // fused with built-in operators in one plan, and differentiated on the device
        same("custom in a graph", () -> SWISH.apply(Tensor.uniform(6, -1, 1, 64, 32).matmul(Tensor.uniform(7, -1, 1, 32, 16))).relu().add(1));
        same("custom gradient", () -> {
            Tensor x = Tensor.uniform(8, -2, 2, 50, 40).requiresGrad(true);
            Tensor w = Tensor.uniform(9, -1, 1, 50, 40).requiresGrad(true);
            HYPOT.apply(SWISH.apply(x), w).mul(Tensor.uniform(10, -1, 1, 50, 40)).sum().backward();
            return x.grad().add(w.grad());
        }, 1e-5, 1e-5);
    }

    @Test
    void printsKernels() {
        JitnnRuntime.reset();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        JitnnRuntime.printKernels(new PrintStream(out, true));
        try {
            SWISH.apply(Tensor.uniform(11, -1, 1, 77, 3)).add(2).toFloatArray();
        } finally {
            JitnnRuntime.printKernels(null);
        }
        String dump = out.toString();
        assertTrue(dump.contains("jitNN plan"), dump);
        assertTrue(dump.contains("custom:test.swish"), dump);
        // the generated source of the user's kernel
        assertTrue(dump.contains("swish"), dump);
        assertTrue(dump.split("swish", -1).length > 2, "no kernel source in:\n" + dump);
        System.out.println(dump);
    }
}
