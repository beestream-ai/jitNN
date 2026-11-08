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

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

/**
 * Reads the MNIST IDX files ({@code train-images-idx3-ubyte[.gz]} and friends) from a directory.
 * Pixels are scaled to [0, 1] and normalised with the usual mean 0.1307 and std 0.3081.
 */
public final class Mnist implements Dataset {

    /** The images [count, 28 * 28], row-major. */
    public final float[] images;
    /** The digit of each image. */
    public final int[] labels;
    /** The number of images. */
    public final int count;
    /** Pixels per image. */
    public static final int PIXELS = 28 * 28;

    private Mnist(float[] images, int[] labels) {
        this.images = images;
        this.labels = labels;
        this.count = labels.length;
    }

    /** Pixels normalised with MNIST's mean and standard deviation (0.1307, 0.3081). */
    public static Mnist load(Path dir, boolean train) throws IOException {
        return load(dir, train, false);
    }

    /** With {@code unit}, pixels are raw / 255 in [0, 1] (targets for generative models); otherwise normalised. */
    public static Mnist load(Path dir, boolean train, boolean unit) throws IOException {
        String prefix = train ? "train" : "t10k";
        float[] images = readImages(open(dir, prefix + "-images-idx3-ubyte"), unit);
        int[] labels = readLabels(open(dir, prefix + "-labels-idx1-ubyte"));
        return new Mnist(images, labels);
    }

    /** Whether {@code dir} holds the MNIST training images (plain or gzipped). */
    public static boolean isPresent(Path dir) {
        return dir != null && (Files.exists(dir.resolve("train-images-idx3-ubyte")) || Files.exists(dir.resolve("train-images-idx3-ubyte.gz")));
    }

    private static DataInputStream open(Path dir, String name) throws IOException {
        Path plain = dir.resolve(name);
        InputStream in = Files.exists(plain) ? Files.newInputStream(plain) : new GZIPInputStream(Files.newInputStream(dir.resolve(name + ".gz")));
        return new DataInputStream(new java.io.BufferedInputStream(in, 1 << 16));
    }

    private static float[] readImages(DataInputStream in, boolean unit) throws IOException {
        try (in) {
            if (in.readInt() != 2051) {
                throw new IOException("not an IDX image file");
            }
            int n = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();
            byte[] raw = in.readNBytes(n * rows * cols);
            float[] out = new float[raw.length];
            for (int i = 0; i < raw.length; i++) {
                float v = (raw[i] & 0xFF) / 255.0f;
                out[i] = unit ? v : (v - 0.1307f) / 0.3081f;
            }
            return out;
        }
    }

    private static int[] readLabels(DataInputStream in) throws IOException {
        try (in) {
            if (in.readInt() != 2049) {
                throw new IOException("not an IDX label file");
            }
            int n = in.readInt();
            byte[] raw = in.readNBytes(n);
            int[] out = new int[n];
            for (int i = 0; i < n; i++) {
                out[i] = raw[i] & 0xFF;
            }
            return out;
        }
    }

    @Override
    public int count() {
        return count;
    }

    @Override
    public int[] sampleShape() {
        return new int[] { 1, 28, 28 };
    }

    /** Copies batch {@code b} (of size {@code size}) of images into {@code x} and labels into {@code y}. */
    @Override
    public void batch(int b, int size, float[] x, int[] y) {
        System.arraycopy(images, b * size * PIXELS, x, 0, size * PIXELS);
        System.arraycopy(labels, b * size, y, 0, size);
    }

    @Override
    public void sample(int i, float[] x, int xAt, int[] y, int yAt) {
        System.arraycopy(images, i * PIXELS, x, xAt, PIXELS);
        y[yAt] = labels[i];
    }
}
