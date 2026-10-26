package org.beehive.jitnn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Random;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.beehive.jitnn.ops.Int8;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * The INT8 products on the device (the tensor-core kernel on CUDA) against the host reference:
 * quantisation and INT32 accumulation are exact, so the results agree up to the FP32 rounding of
 * the dequantisation. Shapes that fill the 128 x 128 tiles and shapes that do not, UINT8 and INT8
 * activations, with and without a bias. Run with {@code -Paccel-tests}.
 */
class Int8AccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static float[] run(Backend backend, int m, int n, int k, boolean unsigned, boolean bias) {
        Random r = new Random(m * 31 + n * 7 + k);
        long[] q = new long[k * n];
        for (int i = 0; i < q.length; i++) {
            q[i] = r.nextInt(255) - 127;
        }
        float[] scales = new float[n];
        for (int i = 0; i < n; i++) {
            scales[i] = 0.001f + r.nextFloat() * 0.01f;
        }
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            b[i] = r.nextFloat() - 0.5f;
        }
        return Engine.with(backend, () -> {
            Int8.Weights w = Int8.Weights.ofColumns(q, k, n, scales);
            Tensor x = Tensor.uniform(3, -2, 2, m, k);
            Tensor scale = Tensor.of(new float[] { 4.0f / 255 }, 1);
            Tensor zp = Tensor.of(new float[] { unsigned ? 128 : 3 }, 1);
            Tensor xq = Int8.quantize(x, scale, zp, unsigned);
            return Int8.linear(xq, scale, zp, unsigned, w, bias ? Tensor.of(b, n) : null).toFloatArray();
        });
    }

    @Test
    void matchesReference() {
        int[][] shapes = { { 128, 128, 32 }, { 256, 384, 768 }, { 37, 70, 50 }, { 1, 9, 3 }, { 300, 130, 129 } };
        for (int[] s : shapes) {
            for (boolean unsigned : new boolean[] { true, false }) {
                for (boolean bias : new boolean[] { false, true }) {
                    float[] e = run(Backend.REFERENCE, s[0], s[1], s[2], unsigned, bias);
                    float[] a = run(Backend.TORNADO, s[0], s[1], s[2], unsigned, bias);
                    assertArrayEquals(e, a, 1e-4f * Math.max(1, s[2] / 32f), "m " + s[0] + " n " + s[1] + " k " + s[2] + (unsigned ? " uint8" : " int8") + (bias ? " bias" : ""));
                }
            }
        }
    }
}
