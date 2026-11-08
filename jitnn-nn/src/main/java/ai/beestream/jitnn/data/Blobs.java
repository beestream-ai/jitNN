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
package ai.beestream.jitnn.data;

import java.util.SplittableRandom;

/** A synthetic classification set: {@code classes} Gaussian clusters in {@code dims} dimensions. */
public final class Blobs {

    /** The samples [count, dims], row-major. */
    public final float[] x;
    /** The cluster of each sample. */
    public final int[] y;
    /** The number of samples. */
    public final int count;
    /** The number of dimensions. */
    public final int dims;

    /**
     * {@code count} samples around {@code classes} centres drawn uniformly in [-2, 2)^dims, with Gaussian
     * noise of standard deviation {@code spread}, from {@code seed}.
     */
    public Blobs(int count, int dims, int classes, float spread, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] centres = new float[classes][dims];
        for (float[] c : centres) {
            for (int d = 0; d < dims; d++) {
                c[d] = (float) (r.nextDouble() * 4 - 2);
            }
        }
        this.x = new float[count * dims];
        this.y = new int[count];
        this.count = count;
        this.dims = dims;
        for (int i = 0; i < count; i++) {
            int c = r.nextInt(classes);
            y[i] = c;
            for (int d = 0; d < dims; d++) {
                x[i * dims + d] = centres[c][d] + (float) (r.nextGaussian() * spread);
            }
        }
    }

    /** Copies batch {@code b} of {@code size} samples into {@code xb} and {@code yb}. */
    public void batch(int b, int size, float[] xb, int[] yb) {
        System.arraycopy(x, b * size * dims, xb, 0, size * dims);
        System.arraycopy(y, b * size, yb, 0, size);
    }
}
