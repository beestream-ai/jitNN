package org.beehive.jitnn.data;

import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Transforms.Transform;

/**
 * Batches of a {@link Dataset} for training, as PyTorch's {@code DataLoader}: shuffled per epoch
 * from a seed, optional per-sample transforms (augmentation), the last short batch kept or dropped,
 * and batches assembled ahead by worker threads while the device computes. Batches come out in
 * order and do not depend on the number of workers: shuffling and transforms draw from random
 * streams derived from the seed, the epoch and the sample.
 *
 * <pre>{@code
 * try (DataLoader loader = DataLoader.builder(train, 128).shuffle(42).transform(Transforms.randomCrop(4)).workers(4).build()) {
 *     for (int epoch = 0; epoch < epochs; epoch++) {
 *         for (DataLoader.Batch b : loader.epoch(epoch)) {
 *             Tensor loss = F.crossEntropy(model.forward(b.x()), b.y());
 *             ...
 *         }
 *     }
 * }
 * }</pre>
 */
public final class DataLoader implements AutoCloseable {

    /** One batch: samples [size, ...sampleShape] and labels [size], host tensors. */
    public record Batch(Tensor x, Tensor y, int size, int index) {
    }

    private final Dataset data;
    private final int batch;
    private final boolean shuffle;
    private final long seed;
    private final boolean dropLast;
    private final Transform transform;
    private final int prefetch;
    private final ExecutorService workers;

    private DataLoader(Builder b) {
        this.data = b.data;
        this.batch = b.batch;
        this.shuffle = b.shuffle;
        this.seed = b.seed;
        this.dropLast = b.dropLast;
        this.transform = b.transform;
        this.prefetch = Math.max(1, b.prefetch);
        this.workers = b.workers > 0 ? Executors.newFixedThreadPool(b.workers, r -> {
            Thread t = new Thread(r, "jitnn-data");
            t.setDaemon(true);
            return t;
        }) : null;
    }

    public static Builder builder(Dataset data, int batchSize) {
        return new Builder(data, batchSize);
    }

    /** Batches per epoch. */
    public int batches() {
        return dropLast ? data.count() / batch : (data.count() + batch - 1) / batch;
    }

    /** The batches of epoch {@code epoch} (its shuffle and augmentation draws). */
    public Iterable<Batch> epoch(int epoch) {
        int[] order = order(epoch);
        return () -> new Iterator<>() {
            private final ArrayDeque<Future<Batch>> ahead = new ArrayDeque<>();
            private int next;
            private int taken;

            @Override
            public boolean hasNext() {
                return taken < batches();
            }

            @Override
            public Batch next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                if (workers == null) {
                    return assemble(order, epoch, taken++);
                }
                while (next < batches() && ahead.size() < prefetch) {
                    int b = next++;
                    ahead.add(workers.submit(() -> assemble(order, epoch, b)));
                }
                try {
                    taken++;
                    return ahead.poll().get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(e);
                } catch (ExecutionException e) {
                    throw new IllegalStateException("loading a batch failed", e.getCause());
                }
            }
        };
    }

    /** The sample order of an epoch: a Fisher-Yates shuffle seeded by (seed, epoch), or dataset order. */
    int[] order(int epoch) {
        int[] o = new int[data.count()];
        for (int i = 0; i < o.length; i++) {
            o[i] = i;
        }
        if (shuffle) {
            SplittableRandom r = new SplittableRandom(seed * 0x9E3779B97F4A7C15L + epoch);
            for (int i = o.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = o[i];
                o[i] = o[j];
                o[j] = t;
            }
        }
        return o;
    }

    private Batch assemble(int[] order, int epoch, int b) {
        int start = b * batch;
        int size = Math.min(batch, order.length - start);
        int sample = data.sampleSize();
        int[] shape = data.sampleShape();
        float[] x = new float[size * sample];
        int[] y = new int[size];
        for (int i = 0; i < size; i++) {
            int s = order[start + i];
            data.sample(s, x, i * sample, y, i);
            if (transform != null) {
                // one stream per (epoch, sample): the draws do not depend on which worker runs it
                transform.apply(x, i * sample, shape, new SplittableRandom(((seed * 31 + epoch) << 32) ^ s));
            }
        }
        int[] dims = new int[shape.length + 1];
        dims[0] = size;
        System.arraycopy(shape, 0, dims, 1, shape.length);
        return new Batch(Tensor.of(x, dims), Tensor.of(y, size), size, b);
    }

    @Override
    public void close() {
        if (workers != null) {
            workers.shutdownNow();
        }
    }

    public static final class Builder {
        private final Dataset data;
        private final int batch;
        private boolean shuffle;
        private long seed;
        private boolean dropLast;
        private Transform transform;
        private int workers;
        private int prefetch = 2;

        private Builder(Dataset data, int batch) {
            if (batch < 1) {
                throw new IllegalArgumentException("batch size " + batch);
            }
            this.data = data;
            this.batch = batch;
        }

        /** Reshuffles every epoch, from {@code seed} and the epoch number. */
        public Builder shuffle(long seed) {
            this.shuffle = true;
            this.seed = seed;
            return this;
        }

        /** Leaves out the last batch when it is short. */
        public Builder dropLast(boolean on) {
            this.dropLast = on;
            return this;
        }

        /** Applied to every sample as its batch is assembled (augmentation, normalisation). */
        public Builder transform(Transform t) {
            this.transform = t;
            return this;
        }

        /** Threads assembling batches ahead (0: on the caller's thread, when asked for). */
        public Builder workers(int n) {
            this.workers = n;
            return this;
        }

        /** Batches assembled ahead of the one being used, with workers (default 2). */
        public Builder prefetch(int batches) {
            this.prefetch = batches;
            return this;
        }

        public DataLoader build() {
            return new DataLoader(this);
        }
    }
}
