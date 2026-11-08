package ai.beestream.jitnn.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ops.Trees;

/**
 * Random forests (Breiman, 2001) and single decision trees on binned features: trees grown on
 * bootstrap samples (weights drawn on the host) with a random subset of the features at each split,
 * predictions averaged. Classification splits on Gini impurity, regression on squared error. A
 * single tree is a forest of one tree without bootstrap or feature sampling, after scikit-learn's
 * DecisionTreeClassifier and DecisionTreeRegressor; thresholds lie on bin edges (maxBins per feature).
 */
public final class RandomForest {

    public enum Task {
        CLASSIFICATION, REGRESSION
    }

    private final Task task;
    private final int trees;
    private final int maxDepth;
    private final int minSamplesLeaf;
    private final int maxFeatures;
    private final int maxBins;
    private final boolean bootstrap;
    private final long seed;
    private Binner binner;
    private Tensor[] packed;
    private int[] classes;
    private int outputs;

    /** maxFeatures <= 0: sqrt(F) for classification, all for regression (scikit-learn's defaults). */
    public RandomForest(Task task, int trees, int maxDepth, int minSamplesLeaf, int maxFeatures, int maxBins, boolean bootstrap, long seed) {
        this.task = task;
        this.trees = trees;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.maxBins = maxBins;
        this.bootstrap = bootstrap;
        this.seed = seed;
    }

    public RandomForest(Task task, int trees, int maxDepth, long seed) {
        this(task, trees, maxDepth, 1, 0, 128, true, seed);
    }

    /** One decision tree. */
    public static RandomForest tree(Task task, int maxDepth, int minSamplesLeaf) {
        return new RandomForest(task, 1, maxDepth, minSamplesLeaf, Integer.MAX_VALUE, 128, false, 0);
    }

    public RandomForest fit(Tensor x, int[] y) {
        if (task != Task.CLASSIFICATION) {
            throw new IllegalStateException("integer labels are for classification");
        }
        classes = Ml.classes(y);
        outputs = classes.length;
        return grow(x, Ml.oneHot(Ml.encode(y, classes), outputs));
    }

    /** Regression targets [n]. */
    public RandomForest fit(Tensor x, Tensor y) {
        if (task != Task.REGRESSION) {
            throw new IllegalStateException("float targets are for regression");
        }
        outputs = 1;
        Tensor y2 = y.reshape(y.dim(0), 1);
        return grow(x, Tensor.cat(1, Tensor.ones(y.dim(0), 1), y2, y2.square()));
    }

    private RandomForest grow(Tensor x, Tensor perSample) {
        Ml.check(x, "RandomForest");
        int n = x.dim(0);
        int f = x.dim(1);
        binner = new Binner(x, maxBins);
        Tensor bins = binner.bin(x);
        int mf = maxFeatures > 0 ? maxFeatures : task == Task.CLASSIFICATION ? Math.max(1, (int) Math.sqrt(f)) : f;
        TreeGrower grower = new TreeGrower(task == Task.CLASSIFICATION ? TreeGrower.Criterion.GINI : TreeGrower.Criterion.MSE, maxDepth, minSamplesLeaf, 0, mf);
        SplittableRandom r = new SplittableRandom(seed);
        List<TreeGrower.Tree> grown = new ArrayList<>();
        for (int t = 0; t < trees; t++) {
            Tensor stats = perSample;
            if (bootstrap) {
                float[] w = new float[n];
                for (int i = 0; i < n; i++) {
                    w[r.nextInt(n)] += 1;
                }
                stats = perSample.mul(Tensor.of(w, n, 1));
            }
            grown.add(grower.grow(bins, binner.bins, stats, outputs, 0, 1.0f / trees, r));
            Ml.step();
        }
        packed = TreeGrower.pack(grown, TreeGrower.maxNodes(maxDepth), outputs);
        return this;
    }

    /** Mean of the trees' leaf values: class probabilities [m, c] or predictions [m, 1]. */
    public Tensor predictValues(Tensor x) {
        return Trees.predict(binner.bin(x), packed[0], packed[1], packed[2], packed[3]);
    }

    public int[] predict(Tensor x) {
        return Ml.decode(predictValues(x).argmax(1).toIntArray(), classes);
    }
}
