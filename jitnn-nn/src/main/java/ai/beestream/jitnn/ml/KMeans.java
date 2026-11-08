package ai.beestream.jitnn.ml;

import java.util.SplittableRandom;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * k-means clustering with Lloyd's algorithm, after scikit-learn's KMeans: k-means++ seeding (or
 * given initial centres), {@code nInit} restarts keeping the lowest inertia, and stopping when the
 * assignments no longer change or the centres move less than tol times the mean feature variance.
 */
public final class KMeans {

    private final int k;
    private final int maxIter;
    private final float tol;
    private final int nInit;
    private final long seed;
    private Tensor centres;
    private float inertia;
    private int iterations;

    /**
     * k-means with {@code k} clusters, at most {@code maxIter} iterations per run, tolerance {@code tol},
     * {@code nInit} runs, seeding from {@code seed}.
     */
    public KMeans(int k, int maxIter, float tol, int nInit, long seed) {
        this.k = k;
        this.maxIter = maxIter;
        this.tol = tol;
        this.nInit = nInit;
        this.seed = seed;
    }

    /** k-means with scikit-learn's defaults: 300 iterations, tolerance 1e-4, one run. */
    public KMeans(int k, long seed) {
        this(k, 300, 1e-4f, 1, seed);
    }

    /** Fits from k-means++ seeds. */
    public KMeans fit(Tensor x) {
        Ml.check(x, "KMeans");
        SplittableRandom r = new SplittableRandom(seed);
        float best = Float.POSITIVE_INFINITY;
        for (int run = 0; run < nInit; run++) {
            Object[] res = lloyd(x, plusPlus(x, r));
            if ((float) res[1] < best) {
                best = (float) res[1];
                centres = (Tensor) res[0];
                inertia = best;
                iterations = (int) res[2];
            }
        }
        return this;
    }

    /** Fits from the given initial centres [k, d]. */
    public KMeans fit(Tensor x, Tensor init) {
        Ml.check(x, "KMeans");
        Object[] res = lloyd(x, init);
        centres = (Tensor) res[0];
        inertia = (float) res[1];
        iterations = (int) res[2];
        return this;
    }

    /** k-means++: each new centre drawn with probability proportional to the squared distance to the nearest chosen one. */
    private Tensor plusPlus(Tensor x, SplittableRandom r) {
        int n = x.dim(0);
        float[] xs = x.toFloatArray();
        int d = x.dim(1);
        int[] chosen = new int[k];
        chosen[0] = r.nextInt(n);
        Tensor closest = Ml.sqDistances(x, x.narrow(0, chosen[0], 1)).reshape(n);
        for (int c = 1; c < k; c++) {
            float[] dist = closest.toFloatArray();
            double total = 0;
            for (float v : dist) {
                total += v;
            }
            double u = r.nextDouble() * total;
            int pick = n - 1;
            for (int i = 0; i < n; i++) {
                u -= dist[i];
                if (u < 0) {
                    pick = i;
                    break;
                }
            }
            chosen[c] = pick;
            closest = closest.minimum(Ml.sqDistances(x, x.narrow(0, pick, 1)).reshape(n));
        }
        float[] init = new float[k * d];
        for (int c = 0; c < k; c++) {
            System.arraycopy(xs, chosen[c] * d, init, c * d, d);
        }
        return Tensor.of(init, k, d);
    }

    /** [centres, inertia, iterations]. */
    private Object[] lloyd(Tensor x, Tensor c) {
        int d = x.dim(1);
        float variance = x.sub(x.mean(0, true)).square().mean().item();
        float tolAbs = tol * variance;
        int[] labels = null;
        int it = 0;
        for (; it < maxIter; it++) {
            Tensor assign = Ml.sqDistances(x, c).neg().argmax(1);
            Tensor oneHot = F.oneHot(assign, k);
            Tensor counts = oneHot.sum(0, false).unsqueeze(1);
            Tensor sums = oneHot.matmul(x, true, false);
            Tensor next = F.where(counts.gt(Tensor.scalar(0.0f)), sums.div(counts.maximum(Tensor.scalar(1.0f))), c);
            float shift = next.sub(c).square().sum().item();
            int[] now = assign.toIntArray();
            c = next;
            if (labels != null && java.util.Arrays.equals(labels, now)) {
                it++;
                break;
            }
            labels = now;
            if (shift <= tolAbs) {
                it++;
                break;
            }
        }
        float in = Ml.sqDistances(x, c).min(1, false).sum().item();
        return new Object[] { c, in, it };
    }

    /** Index of the nearest centre of each row. */
    public int[] predict(Tensor x) {
        return Ml.sqDistances(x, centres).neg().argmax(1).toIntArray();
    }

    /** The cluster centres [k, features]. */
    public Tensor centres() {
        return centres;
    }

    /** Sum of squared distances of the training samples to their nearest centre. */
    public float inertia() {
        return inertia;
    }

    /** The iterations the kept run took. */
    public int iterations() {
        return iterations;
    }
}
