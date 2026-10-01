package org.beehive.jitnn.data;

/** An in-memory labelled dataset of fixed-shape samples (CHW images or flat vectors). */
public interface Dataset {

    int count();

    /** The shape of one sample, e.g. {1, 28, 28} or {3, 32, 32}. */
    int[] sampleShape();

    /** Copies batch {@code b} of {@code size} samples, in dataset order, into {@code x} and {@code y}. */
    void batch(int b, int size, float[] x, int[] y);

    default int sampleSize() {
        int n = 1;
        for (int d : sampleShape()) {
            n *= d;
        }
        return n;
    }
}
