package org.beehive.jitnn.ops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.kernels.RandomKernels;
import org.junit.jupiter.api.Test;

/** Philox4x32-10 against the Random123 known-answer vectors, and the fill statistics. */
class PhiloxTest {

    @Test
    void knownAnswers() {
        assertArrayEquals(new int[] { 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8 }, RandomFill.philox(0, 0, 0, 0, 0, 0));
        assertArrayEquals(new int[] { 0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd }, RandomFill.philox(-1, -1, -1, -1, -1, -1));
        assertArrayEquals(new int[] { 0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1 },
                RandomFill.philox(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0));
    }

    @Test
    void kernelHelpersMatchHost() {
        // The kernel avoids >>> (TornadoVM#1140); its helpers must agree with the host's.
        int[] words = { 0, 1, -1, 0x80000000, 0x7fffffff, 0x6627e8d5, 0xe169c58d };
        for (int a : words) {
            for (int b : words) {
                long p = (a & 0xFFFFFFFFL) * (b & 0xFFFFFFFFL);
                assertEquals((int) (p >>> 32), RandomKernels.mulhi(a, b));
            }
            assertEquals(RandomFill.unit(a), RandomKernels.toUnit(a));
        }
    }

    @Test
    void uniformAndNormalStatistics() {
        Engine.with(Engine.Backend.REFERENCE, () -> {
            float[] u = Tensor.uniform(42, -1, 3, 10_000).toFloatArray();
            float[] z = Tensor.normal(42, 2, 0.5f, 10_001).toFloatArray();
            double su = 0;
            for (float v : u) {
                assertTrue(v >= -1 && v < 3);
                su += v;
            }
            assertEquals(1.0, su / u.length, 0.05);
            double sz = 0;
            double sq = 0;
            for (float v : z) {
                sz += v;
                sq += v * v;
            }
            double mean = sz / z.length;
            assertEquals(2.0, mean, 0.02);
            assertEquals(0.5, Math.sqrt(sq / z.length - mean * mean), 0.02);
            assertArrayEquals(u, Tensor.uniform(42, -1, 3, 10_000).toFloatArray(), "same seed, same values");
            return null;
        });
    }
}
