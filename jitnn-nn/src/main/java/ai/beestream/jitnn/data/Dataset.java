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

/** An in-memory labelled dataset of fixed-shape samples (CHW images or flat vectors). */
public interface Dataset {

    /** The number of samples. */
    int count();

    /** The shape of one sample, e.g. {1, 28, 28} or {3, 32, 32}. */
    int[] sampleShape();

    /** Copies batch {@code b} of {@code size} samples, in dataset order, into {@code x} and {@code y}. */
    void batch(int b, int size, float[] x, int[] y);

    /**
     * Copies sample {@code i} into {@code x} at {@code xAt} and its label into {@code y} at
     * {@code yAt} ({@link DataLoader} assembles shuffled batches with it). The default copies
     * through {@link #batch} of size 1.
     */
    default void sample(int i, float[] x, int xAt, int[] y, int yAt) {
        float[] one = new float[sampleSize()];
        int[] label = new int[1];
        batch(i, 1, one, label);
        System.arraycopy(one, 0, x, xAt, one.length);
        y[yAt] = label[0];
    }

    /** The number of values in one sample: the product of {@link #sampleShape()}. */
    default int sampleSize() {
        int n = 1;
        for (int d : sampleShape()) {
            n *= d;
        }
        return n;
    }

    /** A dataset over arrays: {@code x} holds the samples back to back, {@code y} one label each. */
    static Dataset of(float[] x, int[] y, int... sampleShape) {
        int[] shape = sampleShape.clone();
        int size = x.length / Math.max(y.length, 1);
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n != size || (long) size * y.length != x.length) {
            throw new IllegalArgumentException(x.length + " values and " + y.length + " labels do not make samples of shape " + java.util.Arrays.toString(shape));
        }
        return new Dataset() {
            @Override
            public int count() {
                return y.length;
            }

            @Override
            public int[] sampleShape() {
                return shape.clone();
            }

            @Override
            public void batch(int b, int count, float[] xb, int[] yb) {
                System.arraycopy(x, b * count * size, xb, 0, count * size);
                System.arraycopy(y, b * count, yb, 0, count);
            }

            @Override
            public void sample(int i, float[] xb, int xAt, int[] yb, int yAt) {
                System.arraycopy(x, i * size, xb, xAt, size);
                yb[yAt] = y[i];
            }
        };
    }
}
