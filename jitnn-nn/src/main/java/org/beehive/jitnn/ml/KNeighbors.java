package org.beehive.jitnn.ml;

import org.beehive.jitnn.Tensor;

/**
 * k-nearest neighbours by brute force on the device (Euclidean distance, uniform weights), after
 * scikit-learn's KNeighborsClassifier and KNeighborsRegressor. Class ties go to the smallest label.
 */
public final class KNeighbors {

    private final int k;
    private Tensor x;
    private Tensor targets;
    private int[] classes;

    public KNeighbors(int k) {
        this.k = k;
    }

    /** Classification: integer labels. */
    public KNeighbors fit(Tensor x, int[] y) {
        Ml.check(x, "KNeighbors");
        this.x = x;
        this.classes = Ml.classes(y);
        this.targets = Ml.oneHot(Ml.encode(y, classes), classes.length);
        return this;
    }

    /** Regression: targets [n] or [n, t]. */
    public KNeighbors fit(Tensor x, Tensor y) {
        Ml.check(x, "KNeighbors");
        this.x = x;
        this.classes = null;
        this.targets = y.rank() == 1 ? y.unsqueeze(1) : y;
        return this;
    }

    /** INT32 indices [m, k] of the k nearest training rows of each query row. */
    public Tensor kneighbors(Tensor q) {
        return Ml.sqDistances(q, x).neg().topk(k)[1];
    }

    /** Mean target [m, t] (regression) or class votes [m, c] (classification) of the neighbours. */
    private Tensor neighbourMean(Tensor q) {
        int m = q.dim(0);
        Tensor idx = kneighbors(q).reshape(m * k);
        return targets.indexSelect(0, idx).reshape(m, k, targets.dim(1)).mean(1, false);
    }

    public int[] predict(Tensor q) {
        if (classes == null) {
            throw new IllegalStateException("fitted for regression: use predictValues");
        }
        return Ml.decode(neighbourMean(q).argmax(1).toIntArray(), classes);
    }

    /** Class probabilities [m, c]: the fraction of neighbours per class. */
    public Tensor predictProba(Tensor q) {
        return neighbourMean(q);
    }

    /** Regression predictions [m, t]. */
    public Tensor predictValues(Tensor q) {
        return neighbourMean(q);
    }
}
