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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.function.Supplier;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Grouped 1-D and 3-D convolutions (forward and gradients) against the concatenation of one
 * convolution per group, and max/min/argmax/argmin propagating NaN on every reduction kernel path.
 * Each runs on the device and on the reference backend.
 */
class GroupedConvNaNAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static float[][] both(Supplier<float[][]> f) {
        float[][] dev = Engine.with(Backend.TORNADO, f);
        float[][] ref = Engine.with(Backend.REFERENCE, f);
        for (int i = 0; i < dev.length; i++) {
            close(ref[i], dev[i], "device vs reference, output " + i);
        }
        return ref;
    }

    private static void close(float[] want, float[] got, String what) {
        assertEquals(want.length, got.length, what);
        double scale = 1e-6;
        double err = 0;
        for (int i = 0; i < want.length; i++) {
            assertEquals(Float.isNaN(want[i]), Float.isNaN(got[i]), what + ": NaN at " + i);
            if (!Float.isNaN(want[i])) {
                scale = Math.max(scale, Math.abs(want[i]));
                err = Math.max(err, Math.abs(want[i] - got[i]));
            }
        }
        assertTrue(err <= 1e-4 * scale, what + ": max error " + err + " of " + scale);
    }

    /** Grouped convolution vs per-group convolutions, outputs and the gradients of x and w. */
    private static float[][] grouped(int rank, int groups, boolean split) {
        int n = 2, c = 6, cout = 9, k = 3;
        int[] xd = rank == 1 ? new int[] { n, c, 11 } : new int[] { n, c, 5, 6, 4 };
        int[] wd = rank == 1 ? new int[] { cout, c / groups, k } : new int[] { cout, c / groups, k, k, k };
        Tensor x = Tensor.uniform(1, -1, 1, xd).requiresGrad(true);
        Tensor w = Tensor.uniform(2, -1, 1, wd).requiresGrad(true);
        Tensor b = Tensor.uniform(3, -1, 1, cout);
        Tensor y;
        if (!split) {
            y = rank == 1 ? F.conv1d(x, w, b, 2, 1, 1, groups) : F.conv3d(x, w, b, 1, 1, 1, groups);
        } else {
            Tensor[] parts = new Tensor[groups];
            for (int g = 0; g < groups; g++) {
                Tensor xg = x.narrow(1, g * c / groups, c / groups);
                Tensor wg = w.narrow(0, g * cout / groups, cout / groups);
                Tensor bg = b.narrow(0, g * cout / groups, cout / groups);
                parts[g] = rank == 1 ? F.conv1d(xg, wg, bg, 2, 1, 1) : F.conv3d(xg, wg, bg, 1, 1, 1);
            }
            y = Tensor.cat(1, parts);
        }
        y.square().sum().backward();
        return new float[][] { y.toFloatArray(), x.grad().toFloatArray(), w.grad().toFloatArray() };
    }

    @Test
    void groupedConvolutions() {
        for (int rank : new int[] { 1, 3 }) {
            for (int groups : new int[] { 3 }) {
                float[][] fused = both(() -> grouped(rank, groups, false));
                float[][] split = both(() -> grouped(rank, groups, true));
                for (int i = 0; i < 3; i++) {
                    close(split[i], fused[i], rank + "-D groups " + groups + ", output " + i);
                }
            }
        }
    }

    @Test
    void reductionsPropagateNaN() {
        // lengths that take the per-thread, work-group, split-row and strided kernels
        for (int len : new int[] { 7, 300, 70000 }) {
            int rows = len >= 70000 ? 2 : 6;
            float[] v = new float[rows * len];
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) Math.sin(i * 0.37);
            }
            v[1 * len + len / 2] = Float.NaN;
            v[3 % rows * len + len - 1] = Float.NaN;
            v[3 % rows * len + 1] = Float.NaN;
            int nr = rows;
            float[][] got = both(() -> {
                Tensor x = Tensor.of(v, nr, len);
                Tensor xt = x.transpose(0, 1);
                return new float[][] { x.max(1, false).toFloatArray(), x.min(1, false).toFloatArray(), x.argmax(1).to(DType.F32).toFloatArray(),
                        x.argmin(1).to(DType.F32).toFloatArray(), xt.max(0, false).toFloatArray(), xt.min(0, false).toFloatArray() };
            });
            for (int r = 0; r < rows; r++) {
                boolean nan = r == 1 || r == 3 % rows;
                assertEquals(nan, Float.isNaN(got[0][r]), "max of row " + r + ", length " + len);
                assertEquals(nan, Float.isNaN(got[1][r]), "min of row " + r);
                assertEquals(nan, Float.isNaN(got[4][r]), "strided max of row " + r);
                assertEquals(nan, Float.isNaN(got[5][r]), "strided min of row " + r);
                if (nan) {
                    // row 3 (row 1 when there are two rows) has NaNs at 1 and at the end; row 1 at len / 2
                    int first = r == 3 % rows ? 1 : len / 2;
                    assertEquals(first, (int) got[2][r], "argmax picks the first NaN, row " + r);
                    assertEquals(first, (int) got[3][r], "argmin picks the first NaN, row " + r);
                }
            }
        }
        assertArrayEquals(new float[] { 2 }, Engine.with(Backend.TORNADO, () -> Tensor.of(new float[] { 1, 2, Float.NEGATIVE_INFINITY }, 3).max(0, false).reshape(1).toFloatArray()));
    }
}
