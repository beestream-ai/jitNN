package org.beehive.jitnn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Outputs that outlive their plan's next execution, and outputs released once read
 * ({@link Tensor#release}, try-with-resources): earlier outputs keep their values while the same plan
 * runs again, released ones cannot be read, and pooled host arrays carry no stale values. Run with
 * {@code -Paccel-tests}.
 */
class ReleaseAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static float[] input(int call) {
        float[] v = new float[4096];
        for (int i = 0; i < v.length; i++) {
            v[i] = call * 10000 + i;
        }
        return v;
    }

    private static float[] expected(int call) {
        float[] v = input(call);
        for (int i = 0; i < v.length; i++) {
            v[i] = v[i] * 2 + 1;
        }
        return v;
    }

    /** The same plan each call: x * 2 + 1 over a host input. */
    private static Tensor run(int call) {
        Tensor y = Tensor.of(input(call), 64, 64).mul(2).add(1);
        Engine.sync(y);
        return y;
    }

    @Test
    void earlierOutputsKeepTheirValues() {
        Engine.with(Backend.TORNADO, () -> {
            Tensor[] kept = new Tensor[4];
            for (int c = 0; c < kept.length; c++) {
                kept[c] = run(c);
            }
            for (int c = 0; c < kept.length; c++) {
                assertArrayEquals(expected(c), kept[c].toFloatArray(), "output of call " + c);
                kept[c].release();
            }
            // pooled arrays hold earlier contents: later outputs must not show them
            for (int c = 10; c < 16; c++) {
                Tensor y = run(c);
                Tensor next = run(c + 100);
                assertArrayEquals(expected(c), y.toFloatArray(), "output of call " + c + " after the next call");
                assertArrayEquals(expected(c + 100), next.toFloatArray(), "output of call " + (c + 100));
                y.release();
                next.release();
            }
            return null;
        });
    }

    @Test
    void releasedOutputsCannotBeRead() {
        Engine.with(Backend.TORNADO, () -> {
            Tensor y;
            try (Tensor out = run(1)) {
                assertArrayEquals(expected(1), out.toFloatArray());
                y = out;
            }
            assertThrows(IllegalStateException.class, y::toFloatArray);
            // the plan runs again after the release, and a released output is not copied out first
            try (Tensor out = run(2)) {
                assertArrayEquals(expected(2), out.toFloatArray());
            }
            return null;
        });
    }

    @Test
    void releasingAWeightKeepsItsMeaning() {
        Engine.with(Backend.TORNADO, () -> {
            Tensor w = Tensor.persistent(Tensor.of(input(3), 64, 64));
            Tensor y = w.mul(2).add(1);
            assertArrayEquals(expected(3), y.toFloatArray());
            y.release();
            w.release();
            assertThrows(IllegalStateException.class, () -> w.mul(2).toFloatArray());
            return null;
        });
    }
}
