package ai.beestream.jitnn.data;

import java.util.SplittableRandom;

/** A synthetic classification set: {@code classes} Gaussian clusters in {@code dims} dimensions. */
public final class Blobs {

    public final float[] x;
    public final int[] y;
    public final int count;
    public final int dims;

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

    public void batch(int b, int size, float[] xb, int[] yb) {
        System.arraycopy(x, b * size * dims, xb, 0, size * dims);
        System.arraycopy(y, b * size, yb, 0, size);
    }
}
