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
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ShapeTest {

    @Test
    void broadcasting() {
        assertEquals(Shape.of(4, 3), Shape.broadcast(Shape.of(4, 3), Shape.of(3)));
        assertEquals(Shape.of(2, 4, 3), Shape.broadcast(Shape.of(4, 1), Shape.of(2, 1, 3)));
        assertEquals(Shape.of(5), Shape.broadcast(Shape.scalar(), Shape.of(5)));
        assertThrows(IllegalArgumentException.class, () -> Shape.broadcast(Shape.of(4, 3), Shape.of(4)));
    }

    @Test
    void broadcastStrides() {
        assertArrayEquals(new int[] { 0, 1 }, Shape.of(3).broadcastStrides(Shape.of(4, 3)));
        assertArrayEquals(new int[] { 0, 1, 0 }, Shape.of(4, 1).broadcastStrides(Shape.of(2, 4, 3)));
        assertArrayEquals(new int[] { 12, 3, 1 }, Shape.of(2, 4, 3).strides());
    }

    @Test
    void reduceAndAxes() {
        assertEquals(Shape.of(2, 3), Shape.of(2, 4, 3).reduce(1, false));
        assertEquals(Shape.of(2, 1, 3), Shape.of(2, 4, 3).reduce(-2, true));
        assertEquals(3, Shape.of(2, 4, 3).dim(-1));
        assertEquals(1, Shape.scalar().numel());
    }
}
