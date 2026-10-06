package org.beehive.jitnn.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ops.Trees;

/**
 * Grows one tree level by level: per-node histograms of the sample statistics on the device, the
 * best split of each node on the host, then the samples routed to the children on the device.
 *
 * <p>Statistics per sample (columns of stats): MSE [w, w y, w y^2]; GINI [w 1{y = c}] per class;
 * NEWTON [g, h, w] (gradient boosting). Leaves hold the mean, the class frequencies, or the Newton
 * step -G / (H + lambda).
 */
final class TreeGrower {

    enum Criterion {
        MSE, GINI, NEWTON
    }

    /** A grown tree: per node the split feature (-1 for a leaf), split bin, left child (right = left + 1) and leaf values [nodes, outputs]. */
    static final class Tree {
        final int[] feature;
        final int[] split;
        final int[] child;
        final float[] value;
        final int outputs;

        Tree(int maxNodes, int outputs) {
            feature = new int[maxNodes];
            Arrays.fill(feature, -1);
            split = new int[maxNodes];
            child = new int[maxNodes];
            value = new float[maxNodes * outputs];
            this.outputs = outputs;
        }
    }

    final Criterion criterion;
    final int maxDepth;
    final float minSamplesLeaf;
    final float lambda;
    final int maxFeatures;

    TreeGrower(Criterion criterion, int maxDepth, float minSamplesLeaf, float lambda, int maxFeatures) {
        if (maxDepth < 1 || maxDepth > 16) {
            throw new IllegalArgumentException("tree depth " + maxDepth + " outside 1..16");
        }
        this.criterion = criterion;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.lambda = lambda;
        this.maxFeatures = maxFeatures;
    }

    static int maxNodes(int depth) {
        return (1 << (depth + 1)) - 1;
    }

    /**
     * Grows a tree on bins [n, F] (binCount bins) with stats [n, C]. Leaf values go to column
     * {@code column} of {@code outputs} (NEWTON, scaled by {@code scale}), or fill all outputs.
     */
    Tree grow(Tensor bins, int binCount, Tensor stats, int outputs, int column, float scale, SplittableRandom random) {
        int n = bins.dim(0);
        int features = bins.dim(1);
        int channels = stats.dim(1);
        Tree tree = new Tree(maxNodes(maxDepth), outputs);
        Tensor node = Tensor.of(new int[n], n);
        List<Integer> slots = List.of(0);
        int nextFree = 1;
        for (int depth = 0; depth <= maxDepth && !slots.isEmpty(); depth++) {
            int count = slots.size();
            float[] hist = Trees.histogram(bins, node, stats, count, binCount).toFloatArray();
            int[] feat = new int[count];
            int[] cut = new int[count];
            int[] childSlot = new int[count];
            List<Integer> next = new ArrayList<>();
            for (int s = 0; s < count; s++) {
                int global = slots.get(s);
                double[] total = new double[channels];
                for (int b = 0; b < binCount; b++) {
                    for (int c = 0; c < channels; c++) {
                        total[c] += hist[((s * features) * binCount + b) * channels + c];
                    }
                }
                leafValue(tree, global, total, outputs, column, scale);
                feat[s] = -1;
                if (depth == maxDepth || weight(total) < 2 * minSamplesLeaf) {
                    continue;
                }
                double base = score(total);
                double bestGain = 1e-9 * Math.max(1.0, Math.abs(base));
                int bestF = -1;
                int bestB = -1;
                for (int f : candidates(features, random)) {
                    double[] left = new double[channels];
                    double[] right = new double[channels];
                    for (int b = 0; b + 1 < binCount; b++) {
                        for (int c = 0; c < channels; c++) {
                            left[c] += hist[((s * features + f) * binCount + b) * channels + c];
                            right[c] = total[c] - left[c];
                        }
                        if (weight(left) < minSamplesLeaf || weight(right) < minSamplesLeaf) {
                            continue;
                        }
                        if (criterion == Criterion.NEWTON && (left[1] < 1e-3 || right[1] < 1e-3)) {
                            continue;
                        }
                        double gain = score(left) + score(right) - base;
                        if (gain > bestGain) {
                            bestGain = gain;
                            bestF = f;
                            bestB = b;
                        }
                    }
                }
                if (bestF < 0) {
                    continue;
                }
                feat[s] = bestF;
                cut[s] = bestB;
                childSlot[s] = next.size();
                tree.feature[global] = bestF;
                tree.split[global] = bestB;
                tree.child[global] = nextFree;
                next.add(nextFree);
                next.add(nextFree + 1);
                nextFree += 2;
            }
            if (next.isEmpty()) {
                break;
            }
            node = Trees.route(bins, node, Tensor.of(feat, count), Tensor.of(cut, count), Tensor.of(childSlot, count));
            slots = next;
        }
        return tree;
    }

    private int[] candidates(int features, SplittableRandom random) {
        if (maxFeatures <= 0 || maxFeatures >= features) {
            int[] all = new int[features];
            Arrays.setAll(all, i -> i);
            return all;
        }
        int[] perm = new int[features];
        Arrays.setAll(perm, i -> i);
        for (int i = 0; i < maxFeatures; i++) {
            int j = i + random.nextInt(features - i);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        return Arrays.copyOf(perm, maxFeatures);
    }

    private double weight(double[] t) {
        return switch (criterion) {
            case MSE -> t[0];
            case NEWTON -> t[2];
            case GINI -> Arrays.stream(t).sum();
        };
    }

    /** The node's contribution to the objective, larger is better; a split gains score(L) + score(R) - score(parent). */
    private double score(double[] t) {
        switch (criterion) {
            case MSE:
                return t[0] > 0 ? t[1] * t[1] / t[0] : 0;
            case NEWTON:
                return t[0] * t[0] / (t[1] + lambda);
            default:
                double w = Arrays.stream(t).sum();
                if (w <= 0) {
                    return 0;
                }
                double s = 0;
                for (double v : t) {
                    s += v * v;
                }
                return s / w;
        }
    }

    private void leafValue(Tree tree, int global, double[] t, int outputs, int column, float scale) {
        int at = global * outputs;
        switch (criterion) {
            case MSE -> tree.value[at] = (float) (scale * (t[0] > 0 ? t[1] / t[0] : 0));
            case NEWTON -> tree.value[at + column] = (float) (-scale * t[0] / (t[1] + lambda));
            case GINI -> {
                double w = Arrays.stream(t).sum();
                for (int c = 0; c < outputs; c++) {
                    tree.value[at + c] = (float) (scale * (w > 0 ? t[c] / w : 0));
                }
            }
        }
    }

    /** Device tensors of trees for {@link Trees#predict}: [feature, split, child, value]. */
    static Tensor[] pack(List<Tree> trees, int maxNodes, int outputs) {
        int t = trees.size();
        int[] f = new int[t * maxNodes];
        int[] s = new int[t * maxNodes];
        int[] c = new int[t * maxNodes];
        float[] v = new float[t * maxNodes * outputs];
        for (int i = 0; i < t; i++) {
            Tree tr = trees.get(i);
            System.arraycopy(tr.feature, 0, f, i * maxNodes, maxNodes);
            System.arraycopy(tr.split, 0, s, i * maxNodes, maxNodes);
            System.arraycopy(tr.child, 0, c, i * maxNodes, maxNodes);
            System.arraycopy(tr.value, 0, v, i * maxNodes * outputs, maxNodes * outputs);
        }
        return new Tensor[] { Tensor.of(f, t, maxNodes), Tensor.of(s, t, maxNodes), Tensor.of(c, t, maxNodes), Tensor.of(v, t, maxNodes, outputs) };
    }
}
