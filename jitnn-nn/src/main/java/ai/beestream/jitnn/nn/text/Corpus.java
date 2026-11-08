package ai.beestream.jitnn.nn.text;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Training examples for the embedding models, drawn on the host from sentences of word ids:
 * skip-gram pairs, CBOW windows, negative samples from the unigram distribution raised to 3/4, and
 * GloVe's distance-weighted co-occurrence counts. Everything is driven by a seed.
 */
public final class Corpus {

    private final List<int[]> sentences;
    private final double[] cumulative;
    private final SplittableRandom random;

    /**
     * Examples from {@code sentences} of word ids, negatives drawn by the word counts {@code counts}
     * (indexed by id), randomness from {@code seed}.
     */
    public Corpus(List<int[]> sentences, long[] counts, long seed) {
        this.sentences = sentences;
        this.random = new SplittableRandom(seed);
        cumulative = new double[counts.length];
        double total = 0;
        for (int i = 0; i < counts.length; i++) {
            total += Math.pow(counts[i], 0.75);
            cumulative[i] = total;
        }
        for (int i = 0; i < counts.length; i++) {
            cumulative[i] /= total;
        }
    }

    /** The sentences, as word ids. */
    public List<int[]> sentences() {
        return sentences;
    }

    /** {@code n} words drawn from the unigram distribution to the power 3/4. */
    public int[] negatives(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            int at = Arrays.binarySearch(cumulative, random.nextDouble());
            out[i] = Math.min(at < 0 ? -at - 1 : at, cumulative.length - 1);
        }
        return out;
    }

    /**
     * Skip-gram (centre, context) pairs with word2vec's dynamic window (each centre draws a window
     * 1..window), shuffled: [centres, contexts]. With {@code docs} the centre is the sentence index
     * instead (PV-DBOW).
     */
    public int[][] skipGramPairs(int window, boolean docs) {
        List<int[]> pairs = new ArrayList<>();
        for (int s = 0; s < sentences.size(); s++) {
            int[] t = sentences.get(s);
            for (int i = 0; i < t.length; i++) {
                if (docs) {
                    pairs.add(new int[] { s, t[i] });
                    continue;
                }
                int w = 1 + random.nextInt(window);
                for (int j = Math.max(0, i - w); j <= Math.min(t.length - 1, i + w); j++) {
                    if (j != i) {
                        pairs.add(new int[] { t[i], t[j] });
                    }
                }
            }
        }
        return split(shuffle(pairs), 2);
    }

    /**
     * CBOW samples, only positions with a full window on both sides, shuffled: [sentence, centre,
     * context (2 window per sample, flattened)].
     */
    public int[][] cbowSamples(int window) {
        List<int[]> samples = new ArrayList<>();
        for (int s = 0; s < sentences.size(); s++) {
            int[] t = sentences.get(s);
            for (int i = window; i + window < t.length; i++) {
                int[] row = new int[2 + 2 * window];
                row[0] = s;
                row[1] = t[i];
                int k = 2;
                for (int j = i - window; j <= i + window; j++) {
                    if (j != i) {
                        row[k++] = t[j];
                    }
                }
                samples.add(row);
            }
        }
        shuffle(samples);
        int n = samples.size();
        int[] sent = new int[n];
        int[] centre = new int[n];
        int[] ctx = new int[n * 2 * window];
        for (int i = 0; i < n; i++) {
            int[] r = samples.get(i);
            sent[i] = r[0];
            centre[i] = r[1];
            System.arraycopy(r, 2, ctx, i * 2 * window, 2 * window);
        }
        return new int[][] { sent, centre, ctx };
    }

    /** Symmetric co-occurrence counts within {@code window}, each weighted 1 / distance: [rows, cols, value]. */
    public Object[] cooccurrence(int window) {
        Map<Long, Double> x = new HashMap<>();
        for (int[] t : sentences) {
            for (int i = 0; i < t.length; i++) {
                for (int j = Math.max(0, i - window); j < i; j++) {
                    double w = 1.0 / (i - j);
                    x.merge(((long) t[i] << 32) | t[j], w, Double::sum);
                    x.merge(((long) t[j] << 32) | t[i], w, Double::sum);
                }
            }
        }
        int n = x.size();
        int[] r = new int[n];
        int[] c = new int[n];
        float[] v = new float[n];
        int k = 0;
        for (var e : x.entrySet()) {
            r[k] = (int) (e.getKey() >>> 32);
            c[k] = (int) (long) e.getKey();
            v[k] = (float) (double) e.getValue();
            k++;
        }
        return new Object[] { r, c, v };
    }

    private List<int[]> shuffle(List<int[]> rows) {
        for (int i = rows.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int[] t = rows.get(i);
            rows.set(i, rows.get(j));
            rows.set(j, t);
        }
        return rows;
    }

    private static int[][] split(List<int[]> rows, int width) {
        int[][] out = new int[width][rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            for (int k = 0; k < width; k++) {
                out[k][i] = rows.get(i)[k];
            }
        }
        return out;
    }

    /** Cosine similarity of rows a and b of vectors [n, dim]. */
    public static float cosine(float[] vectors, int dim, int a, int b) {
        double dot = 0, na = 0, nb = 0;
        for (int d = 0; d < dim; d++) {
            float x = vectors[a * dim + d];
            float y = vectors[b * dim + d];
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        return (float) (dot / Math.sqrt(Math.max(na * nb, 1e-30)));
    }

    /** The row of vectors [n, dim] most similar to row {@code of} (by cosine), excluding itself. */
    public static int nearest(float[] vectors, int dim, int of) {
        int n = vectors.length / dim;
        int best = -1;
        float bestSim = -2;
        for (int i = 0; i < n; i++) {
            if (i != of) {
                float s = cosine(vectors, dim, of, i);
                if (s > bestSim) {
                    bestSim = s;
                    best = i;
                }
            }
        }
        return best;
    }
}
