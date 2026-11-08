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
import static ai.beestream.jitnn.Checks.gradCheck;
import static ai.beestream.jitnn.Checks.seq;
import static ai.beestream.jitnn.CustomOpFixtures.HYPOT;
import static ai.beestream.jitnn.CustomOpFixtures.MIN_MAX;
import static ai.beestream.jitnn.CustomOpFixtures.REPEAT_SCALED;
import static ai.beestream.jitnn.CustomOpFixtures.SWISH;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.Test;

/** Custom operators on the reference backend: values, attributes, several outputs, gradients and errors. */
class CustomOpTest {

    @Test
    void unaryAndBinary() {
        Engine.with(Backend.REFERENCE, () -> {
            float[] x = seq(6, 1.3f, -2f);
            float[] y = SWISH.apply(Tensor.of(x, 2, 3)).toFloatArray();
            float[] expected = new float[6];
            for (int i = 0; i < 6; i++) {
                expected[i] = (float) (x[i] / (1 + Math.exp(-x[i])));
            }
            close(expected, y, 1e-6, 1e-7, "swish");
            float[] h = HYPOT.apply(Tensor.of(new float[] { 3, 5 }, 2), Tensor.of(new float[] { 4, 12 }, 2)).toFloatArray();
            assertArrayEquals(new float[] { 5, 13 }, h, 1e-6f);
            return null;
        });
    }

    @Test
    void severalOutputsAndAttributes() {
        Engine.with(Backend.REFERENCE, () -> {
            Tensor[] mm = MIN_MAX.applyAll(null, null, Tensor.of(new float[] { 3, -1, 2, 7, 0, 5 }, 2, 3));
            assertArrayEquals(new float[] { -1, 0 }, mm[0].toFloatArray());
            assertArrayEquals(new float[] { 3, 7 }, mm[1].toFloatArray());
            float[] r = REPEAT_SCALED.apply(new int[] { 3 }, new float[] { 2 }, Tensor.of(new float[] { 1, 2 }, 2)).toFloatArray();
            assertArrayEquals(new float[] { 2, 4, 2, 4, 2, 4 }, r);
            return null;
        });
    }

    @Test
    void gradients() {
        gradCheck("custom swish", x -> SWISH.apply(x[0]).square().sum(), new float[][] { seq(6, 1.3f, -2f) }, new int[][] { { 2, 3 } });
        gradCheck("custom hypot", x -> HYPOT.apply(x[0], x[1]).sum(), new float[][] { seq(6, 0.7f, 0.5f), seq(6, -0.9f, 1.1f) }, new int[][] { { 6 }, { 6 } });
        gradCheck("custom repeat", x -> REPEAT_SCALED.apply(new int[] { 3 }, new float[] { 1.5f }, x[0]).square().sum(), new float[][] { seq(4, 0.6f, -1f) },
                new int[][] { { 2, 2 } });
    }

    @Test
    void errors() {
        assertThrows(IllegalArgumentException.class, () -> CustomOp.unary("test.swish", CustomOpFixtures::swish, v -> v));
        assertThrows(IllegalArgumentException.class, () -> HYPOT.apply(Tensor.zeros(2), Tensor.zeros(3)));
        assertThrows(IllegalArgumentException.class, () -> SWISH.apply(Tensor.of(new int[] { 1 }, 1)));
        assertThrows(IllegalStateException.class, () -> CustomOp.builder("test.incomplete").inputs(DType.F32).build());
        Exception reserved = assertThrows(IllegalArgumentException.class, () -> CustomOp.unary("test.exp", CustomOpFixtures::exp, Math::exp));
        assertTrue(reserved.getMessage().contains("rename"), reserved.getMessage());
        CustomOp noHost = CustomOp.unary("test.deviceOnly", CustomOpFixtures::swish, null);
        Exception e = assertThrows(UnsupportedOperationException.class, () -> Engine.with(Backend.REFERENCE, () -> noHost.apply(Tensor.zeros(2)).toFloatArray()));
        assertTrue(e.getMessage().contains("no host reference"));
        CustomOp noGrad = CustomOp.unary("test.noGradient", CustomOpFixtures::swish, v -> v);
        assertThrows(UnsupportedOperationException.class,
                () -> Engine.with(Backend.REFERENCE, () -> {
                    Tensor x = Tensor.zeros(2).requiresGrad(true);
                    noGrad.apply(x).sum().backward();
                    return null;
                }));
    }
}
