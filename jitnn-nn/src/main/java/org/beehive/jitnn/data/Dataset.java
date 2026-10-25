package org.beehive.jitnn.data;

/** An in-memory labelled dataset of fixed-shape samples (CHW images or flat vectors). */
public interface Dataset {

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
