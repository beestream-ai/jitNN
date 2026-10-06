package org.beehive.jitnn.ml;

import org.beehive.jitnn.Tensor;

/**
 * Naive Bayes classifiers after scikit-learn's GaussianNB (per-class means and variances, the
 * variances smoothed by varSmoothing times the largest feature variance) and MultinomialNB
 * (feature counts with additive smoothing alpha). Class priors are the class frequencies.
 */
public final class NaiveBayes {

    public enum Kind {
        GAUSSIAN, MULTINOMIAL
    }

    private final Kind kind;
    private final float smoothing;
    private int[] classes;
    private Tensor logPrior;
    private Tensor theta;
    private Tensor var;
    private Tensor featureLogProb;

    /** smoothing: varSmoothing (Gaussian, scikit-learn's 1e-9) or alpha (multinomial, 1). */
    public NaiveBayes(Kind kind, float smoothing) {
        this.kind = kind;
        this.smoothing = smoothing;
    }

    public NaiveBayes fit(Tensor x, int[] y) {
        Ml.check(x, "NaiveBayes");
        classes = Ml.classes(y);
        int c = classes.length;
        Tensor oh = Ml.oneHot(Ml.encode(y, classes), c);
        Tensor counts = oh.sum(0, false);
        logPrior = counts.mul(1.0f / x.dim(0)).log();
        Tensor sums = oh.matmul(x, true, false);
        if (kind == Kind.GAUSSIAN) {
            Tensor n = counts.unsqueeze(1);
            theta = sums.div(n);
            Tensor sq = oh.matmul(x.square(), true, false).div(n);
            float eps = smoothing * x.sub(x.mean(0, true)).square().mean(0, false).max(0, false).item();
            var = sq.sub(theta.square()).relu().add(eps);
        } else {
            Tensor smoothed = sums.add(smoothing);
            featureLogProb = smoothed.log().sub(smoothed.sum(1, true).log());
        }
        return this;
    }

    /** Joint log-likelihood [m, c]. */
    public Tensor jointLogLikelihood(Tensor x) {
        if (kind == Kind.MULTINOMIAL) {
            return x.matmul(featureLogProb, false, true).add(logPrior);
        }
        // -0.5 sum log(2 pi var) - 0.5 sum (x - theta)^2 / var
        Tensor inv = var.reciprocal();
        Tensor quad = x.square().matmul(inv, false, true).sub(x.matmul(theta.mul(inv), false, true).mul(2.0f)).add(theta.square().mul(inv).sum(1, false));
        Tensor norm = var.mul((float) (2 * Math.PI)).log().sum(1, false);
        return quad.add(norm).mul(-0.5f).add(logPrior);
    }

    public Tensor predictProba(Tensor x) {
        return jointLogLikelihood(x).softmax();
    }

    public int[] predict(Tensor x) {
        return Ml.decode(jointLogLikelihood(x).argmax(1).toIntArray(), classes);
    }

    public Tensor theta() {
        return theta;
    }

    public Tensor var() {
        return var;
    }

    public Tensor featureLogProb() {
        return featureLogProb;
    }
}
