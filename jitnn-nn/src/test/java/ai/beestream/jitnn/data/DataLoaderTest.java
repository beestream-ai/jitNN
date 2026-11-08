package ai.beestream.jitnn.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class DataLoaderTest {

    @BeforeAll
    static void reference() {
        Engine.setBackend(Engine.Backend.REFERENCE);
    }

    /** 50 samples of shape [2, 4, 4]: sample i holds i * 1000 + position, label i. */
    private static Dataset numbered(int count) {
        float[] x = new float[count * 32];
        int[] y = new int[count];
        for (int i = 0; i < count; i++) {
            y[i] = i;
            for (int k = 0; k < 32; k++) {
                x[i * 32 + k] = i * 1000 + k;
            }
        }
        return Dataset.of(x, y, 2, 4, 4);
    }

    private static List<float[]> collect(DataLoader l, int epoch, List<int[]> labels) {
        List<float[]> xs = new ArrayList<>();
        for (DataLoader.Batch b : l.epoch(epoch)) {
            xs.add(b.x().toFloatArray());
            labels.add(b.y().toIntArray());
        }
        return xs;
    }

    @Test
    void shufflesEveryEpochAndCoversTheData() {
        try (DataLoader l = DataLoader.builder(numbered(50), 8).shuffle(7).build()) {
            assertEquals(7, l.batches());
            List<int[]> e0 = new ArrayList<>();
            List<int[]> e1 = new ArrayList<>();
            collect(l, 0, e0);
            collect(l, 1, e1);
            int[] all0 = e0.stream().flatMapToInt(Arrays::stream).sorted().toArray();
            int[] expected = new int[50];
            Arrays.setAll(expected, i -> i);
            assertArrayEquals(expected, all0);
            assertEquals(2, e0.get(6).length);
            assertFalse(Arrays.equals(e0.stream().flatMapToInt(Arrays::stream).toArray(), e1.stream().flatMapToInt(Arrays::stream).toArray()), "epochs shuffled alike");
        }
        try (DataLoader l = DataLoader.builder(numbered(50), 8).dropLast(true).build()) {
            List<int[]> e = new ArrayList<>();
            collect(l, 0, e);
            assertEquals(6, e.size());
            assertArrayEquals(new int[] { 8, 9, 10, 11, 12, 13, 14, 15 }, e.get(1));
        }
    }

    @Test
    void workersGiveTheSameBatches() {
        Transforms.Transform aug = Transforms.randomCrop(1).then(Transforms.randomHorizontalFlip()).then(Transforms.normalize(new float[] { 1, 2 }, new float[] { 3, 4 }));
        List<int[]> y0 = new ArrayList<>();
        List<int[]> y4 = new ArrayList<>();
        List<float[]> x0;
        List<float[]> x4;
        try (DataLoader l = DataLoader.builder(numbered(50), 8).shuffle(3).transform(aug).build()) {
            x0 = collect(l, 2, y0);
        }
        try (DataLoader l = DataLoader.builder(numbered(50), 8).shuffle(3).transform(aug).workers(4).prefetch(3).build()) {
            x4 = collect(l, 2, y4);
        }
        assertEquals(x0.size(), x4.size());
        for (int i = 0; i < x0.size(); i++) {
            assertArrayEquals(x0.get(i), x4.get(i), "batch " + i);
            assertArrayEquals(y0.get(i), y4.get(i), "labels " + i);
        }
    }

    @Test
    void transforms() {
        int[] shape = { 1, 2, 3 };
        float[] x = { 1, 2, 3, 4, 5, 6 };
        // a stream whose first boolean is true flips
        SplittableRandom flips = null;
        for (long s = 0; flips == null; s++) {
            if (new SplittableRandom(s).nextBoolean()) {
                flips = new SplittableRandom(s);
            }
        }
        Transforms.randomHorizontalFlip().apply(x, 0, shape, flips);
        assertArrayEquals(new float[] { 3, 2, 1, 6, 5, 4 }, x);
        float[] n = { 1, 3, 5, 7 };
        Transforms.normalize(new float[] { 1, 3 }, new float[] { 2, 4 }).apply(n, 0, new int[] { 2, 1, 2 }, null);
        assertArrayEquals(new float[] { 0, 1, 0.5f, 1 }, n);
        // a crop keeps the values, shifted, and fills with zeros
        float[] c = new float[16];
        for (int i = 0; i < c.length; i++) {
            c[i] = i + 1;
        }
        Transforms.randomCrop(1).apply(c, 0, new int[] { 1, 4, 4 }, new SplittableRandom(5));
        int kept = 0;
        for (float v : c) {
            kept += v != 0 ? 1 : 0;
        }
        assertTrue(kept >= 9, "crop kept " + kept);
    }

    @Test
    void prefetchOverlapsLoadingWithWork() {
        Dataset slow = new Dataset() {
            final Dataset d = numbered(80);

            @Override
            public int count() {
                return d.count();
            }

            @Override
            public int[] sampleShape() {
                return d.sampleShape();
            }

            @Override
            public void batch(int b, int size, float[] x, int[] y) {
                d.batch(b, size, x, y);
            }

            @Override
            public void sample(int i, float[] x, int xAt, int[] y, int yAt) {
                sleep(2);
                d.sample(i, x, xAt, y, yAt);
            }
        };
        long[] ms = new long[2];
        for (int w = 0; w < 2; w++) {
            try (DataLoader l = DataLoader.builder(slow, 8).workers(w * 2).build()) {
                long t0 = System.nanoTime();
                for (DataLoader.Batch b : l.epoch(0)) {
                    sleep(16);
                }
                ms[w] = (System.nanoTime() - t0) / 1_000_000;
            }
        }
        assertTrue(ms[1] < 0.75 * ms[0], "with workers " + ms[1] + " ms, without " + ms[0] + " ms");
    }

    private static void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
