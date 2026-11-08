package ai.beestream.jitnn.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ops.Trees;

/**
 * Gradient-boosted trees with second-order (Newton) leaves on binned features, after
 * scikit-learn's HistGradientBoosting and XGBoost: squared error for regression, the logistic
 * loss for two classes and the softmax loss (one tree per class and round) for more. Gradients
 * are computed on the device each round; the trees are grown as in {@link RandomForest}.
 */
public final class GradientBoosting {

    private final boolean classification;
    private final int rounds;
    private final float learningRate;
    private final int maxDepth;
    private final int minSamplesLeaf;
    private final float lambda;
    private final int maxBins;
    private Binner binner;
    private Tensor[] packed;
    private float[] baseline;
    private int[] classes;
    private int outputs;

    /**
     * Boosting for classification or regression: {@code rounds} trees of depth at most {@code maxDepth},
     * shrunk by {@code learningRate}, with at least {@code minSamplesLeaf} samples per leaf, L2 penalty
     * {@code lambda} on the leaf values, and features in at most {@code maxBins} bins.
     */
    public GradientBoosting(boolean classification, int rounds, float learningRate, int maxDepth, int minSamplesLeaf, float lambda, int maxBins) {
        this.classification = classification;
        this.rounds = rounds;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.lambda = lambda;
        this.maxBins = maxBins;
    }

    /** Boosting with scikit-learn's defaults for the rest: 20 samples per leaf, no L2 penalty, 255 bins. */
    public GradientBoosting(boolean classification, int rounds, float learningRate, int maxDepth) {
        this(classification, rounds, learningRate, maxDepth, 20, 0.0f, 255);
    }

    /** Fits a classifier to samples x [n, features] and labels y. */
    public GradientBoosting fit(Tensor x, int[] y) {
        if (!classification) {
            throw new IllegalStateException("integer labels are for classification");
        }
        classes = Ml.classes(y);
        int[] idx = Ml.encode(y, classes);
        int k = classes.length == 2 ? 1 : classes.length;
        outputs = k;
        baseline = new float[k];
        int n = idx.length;
        if (k == 1) {
            double p = 0;
            for (int v : idx) {
                p += v;
            }
            p /= n;
            baseline[0] = (float) Math.log(p / (1 - p));
            float[] t = new float[n];
            for (int i = 0; i < n; i++) {
                t[i] = idx[i];
            }
            return boost(x, Tensor.of(t, n, 1));
        }
        double[] count = new double[k];
        for (int v : idx) {
            count[v]++;
        }
        for (int c = 0; c < k; c++) {
            baseline[c] = (float) Math.log(Math.max(count[c], 1) / n);
        }
        return boost(x, Ml.oneHot(idx, k));
    }

    /** Regression targets [n]. */
    public GradientBoosting fit(Tensor x, Tensor y) {
        if (classification) {
            throw new IllegalStateException("float targets are for regression");
        }
        outputs = 1;
        baseline = new float[] { y.mean().item() };
        return boost(x, y.reshape(y.dim(0), 1));
    }

    private GradientBoosting boost(Tensor x, Tensor target) {
        Ml.check(x, "GradientBoosting");
        int n = x.dim(0);
        binner = new Binner(x, maxBins);
        Tensor bins = binner.bin(x);
        TreeGrower grower = new TreeGrower(TreeGrower.Criterion.NEWTON, maxDepth, minSamplesLeaf, lambda, 0);
        SplittableRandom r = new SplittableRandom(0);
        Tensor raw = Tensor.of(baseline, 1, outputs).expand(ai.beestream.jitnn.Shape.of(n, outputs)).mul(1.0f);
        Tensor ones = Tensor.ones(n, 1);
        List<TreeGrower.Tree> grown = new ArrayList<>();
        int maxNodes = TreeGrower.maxNodes(maxDepth);
        for (int round = 0; round < rounds; round++) {
            Tensor g;
            Tensor h;
            if (!classification) {
                g = raw.sub(target);
                h = Tensor.ones(n, 1);
            } else if (outputs == 1) {
                Tensor p = raw.sigmoid();
                g = p.sub(target);
                h = p.mul(p.neg().add(1.0f));
            } else {
                Tensor p = raw.softmax();
                g = p.sub(target);
                h = p.mul(p.neg().add(1.0f));
            }
            Ml.step();
            List<TreeGrower.Tree> roundTrees = new ArrayList<>();
            for (int c = 0; c < outputs; c++) {
                Tensor stats = Tensor.cat(1, g.narrow(1, c, 1), h.narrow(1, c, 1), ones);
                roundTrees.add(grower.grow(bins, binner.bins, stats, outputs, c, learningRate, r));
            }
            grown.addAll(roundTrees);
            Tensor[] t = TreeGrower.pack(roundTrees, maxNodes, outputs);
            raw = raw.add(Trees.predict(bins, t[0], t[1], t[2], t[3]));
            Ml.step();
        }
        packed = TreeGrower.pack(grown, maxNodes, outputs);
        return this;
    }

    /** Raw scores [m, outputs]: the baseline plus every tree. */
    public Tensor decisionFunction(Tensor x) {
        return Trees.predict(binner.bin(x), packed[0], packed[1], packed[2], packed[3]).add(Tensor.of(baseline, 1, outputs));
    }

    /** Regression predictions [m]. */
    public Tensor predictValues(Tensor x) {
        return decisionFunction(x).reshape(x.dim(0));
    }

    /** Class probabilities [n, classes] for samples x. */
    public Tensor predictProba(Tensor x) {
        Tensor z = decisionFunction(x);
        if (outputs == 1) {
            Tensor p1 = z.sigmoid();
            return Tensor.cat(1, p1.neg().add(1.0f), p1);
        }
        return z.softmax();
    }

    /** The most probable label of each sample. */
    public int[] predict(Tensor x) {
        return Ml.decode(predictProba(x).argmax(1).toIntArray(), classes);
    }
}
