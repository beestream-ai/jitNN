package ai.beestream.jitnn.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * CIFAR-10 from the binary distribution ({@code cifar-10-batches-bin}: data_batch_1..5.bin and
 * test_batch.bin, each record a label byte and 3072 pixel bytes in CHW order). Pixels are scaled
 * to [0, 1] and normalised per channel with the usual means (0.4914, 0.4822, 0.4465) and standard
 * deviations (0.2470, 0.2435, 0.2616). No augmentation: batches come in file order.
 */
public final class Cifar10 implements Dataset {

    public static final int PIXELS = 3 * 32 * 32;
    private static final float[] MEAN = { 0.4914f, 0.4822f, 0.4465f };
    private static final float[] STD = { 0.2470f, 0.2435f, 0.2616f };

    public final float[] images;
    public final int[] labels;
    public final int count;

    private Cifar10(float[] images, int[] labels) {
        this.images = images;
        this.labels = labels;
        this.count = labels.length;
    }

    /** {@code dir} is the directory holding cifar-10-batches-bin (or that directory itself). */
    public static Cifar10 load(Path dir, boolean train) throws IOException {
        Path base = Files.isDirectory(dir.resolve("cifar-10-batches-bin")) ? dir.resolve("cifar-10-batches-bin") : dir;
        String[] files = train ? new String[] { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" }
                : new String[] { "test_batch.bin" };
        int records = files.length * 10_000;
        float[] images = new float[records * PIXELS];
        int[] labels = new int[records];
        int r = 0;
        for (String f : files) {
            byte[] raw = Files.readAllBytes(base.resolve(f));
            if (raw.length != 10_000 * (PIXELS + 1)) {
                throw new IOException(f + " is not a CIFAR-10 binary batch");
            }
            for (int i = 0; i < 10_000; i++, r++) {
                int at = i * (PIXELS + 1);
                labels[r] = raw[at] & 0xFF;
                for (int p = 0; p < PIXELS; p++) {
                    int c = p / 1024;
                    images[r * PIXELS + p] = ((raw[at + 1 + p] & 0xFF) / 255.0f - MEAN[c]) / STD[c];
                }
            }
        }
        return new Cifar10(images, labels);
    }

    public static boolean isPresent(Path dir) {
        return dir != null && (Files.exists(dir.resolve("cifar-10-batches-bin/test_batch.bin")) || Files.exists(dir.resolve("test_batch.bin")));
    }

    @Override
    public int count() {
        return count;
    }

    @Override
    public int[] sampleShape() {
        return new int[] { 3, 32, 32 };
    }

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
