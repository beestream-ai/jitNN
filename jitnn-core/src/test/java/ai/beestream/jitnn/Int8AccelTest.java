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

import java.util.Random;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import ai.beestream.jitnn.ops.Int8;
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

    /** [N, C, H, W, Cout, k, stride, pad] */
    private static final int[][] CONVS = { { 2, 3, 17, 17, 20, 3, 2, 1 }, { 1, 64, 14, 14, 64, 3, 1, 1 }, { 3, 256, 7, 7, 64, 1, 1, 0 }, { 1, 8, 9, 11, 130, 5, 2, 2 } };

    private static float[] conv(Backend backend, int[] c, boolean unsigned, boolean dequantised) {
        int n = c[0], ch = c[1], h = c[2], wd = c[3], cout = c[4], k = c[5];
        Random r = new Random(c[1] * 13 + c[4]);
        long[] q = new long[cout * ch * k * k];
        float[] wf = new float[q.length];
        float[] scales = new float[cout];
        for (int o = 0; o < cout; o++) {
            scales[o] = 0.001f + r.nextFloat() * 0.01f;
        }
        for (int i = 0; i < q.length; i++) {
            q[i] = r.nextInt(255) - 127;
            wf[i] = q[i] * scales[i / (ch * k * k)];
        }
        float[] b = new float[cout];
        for (int i = 0; i < cout; i++) {
            b[i] = r.nextFloat() - 0.5f;
        }
        float s = 4.0f / 255;
        float z = unsigned ? 128 : 3;
        return Engine.with(backend, () -> {
            Tensor x = Tensor.uniform(5, -2, 2, n, ch, h, wd);
            int[] stride = { c[6], c[6] };
            int[] pads = { c[7], c[7], c[7], c[7] };
            int[] dil = { 1, 1 };
            if (dequantised) {
                // the float convolution of the fake-quantised input and the dequantised weights
                Tensor fq = x.div(Tensor.scalar(s)).round().add(Tensor.scalar(z)).clamp(unsigned ? 0 : -128, unsigned ? 255 : 127).sub(Tensor.scalar(z)).mul(Tensor.scalar(s));
                return F.conv(fq, Tensor.of(wf, cout, ch, k, k), Tensor.of(b, cout), stride, pads, dil, 1).toFloatArray();
            }
            Int8.Weights w = Int8.Weights.ofConv(q, cout, ch, k, k, scales);
            Tensor scale = Tensor.of(new float[] { s }, 1);
            Tensor zp = Tensor.of(new float[] { z }, 1);
            return Int8.conv2d(x, scale, zp, unsigned, w, Tensor.of(b, cout), k, k, stride, pads, dil).toFloatArray();
        });
    }

    @Test
    void convolutionsMatchReference() {
        for (int[] c : CONVS) {
            for (boolean unsigned : new boolean[] { true, false }) {
                String name = java.util.Arrays.toString(c) + (unsigned ? " uint8" : " int8");
                float[] e = conv(Backend.REFERENCE, c, unsigned, false);
                assertArrayEquals(conv(Backend.REFERENCE, c, unsigned, true), e, 1e-3f, name + ": INT8 against the dequantised float convolution");
                assertArrayEquals(e, conv(Backend.TORNADO, c, unsigned, false), 1e-3f, name);
            }
        }
    }
}
