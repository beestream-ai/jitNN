package org.beehive.jitnn;

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
