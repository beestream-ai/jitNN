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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Tensor methods the modules use instead of internals, on the reference backend. */
class TensorApiTest {

    @Test
    void assignCopiesIntoAPersistentTensor() {
        JitnnRuntime.with(Backend.REFERENCE, () -> {
            Tensor w = Tensor.persistent(Tensor.of(new float[] { 1, 2, 3 }, 3));
            assertTrue(w.isPersistent());
            w.assign(w.mul(2).add(1));
            assertArrayEquals(new float[] { 3, 5, 7 }, w.toFloatArray());
            Tensor plain = Tensor.of(new float[] { 1, 2, 3 }, 3);
            assertFalse(plain.isPersistent());
            assertThrows(IllegalArgumentException.class, () -> plain.assign(w));
            return null;
        });
    }

    @Test
    void leavesAndPowers() {
        JitnnRuntime.with(Backend.REFERENCE, () -> {
            Tensor x = Tensor.of(new float[] { 1, 2, 3 }, 3).requiresGrad(true);
            Tensor y = x.pow(Tensor.of(new float[] { 2 }, 1));
            assertTrue(x.isLeaf());
            assertFalse(y.isLeaf());
            assertArrayEquals(new float[] { 1, 4, 9 }, y.toFloatArray(), 1e-6f);
            assertArrayEquals(new float[] { 0.5f, 1, 1.5f }, F.fakeQuantize(Tensor.of(new float[] { 0.4f, 1.1f, 1.6f }, 3), 0.5f, -128, 8).toFloatArray(), 1e-6f);
            return null;
        });
    }
}
