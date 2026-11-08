package ai.beestream.jitnn.ml;

import java.util.Arrays;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ops.Trees;

/**
 * Quantises features into at most maxBins bins per feature, as scikit-learn's HistGradientBoosting
 * does: features with few distinct values split between them, others at quantiles. A value goes
 * to the first bin whose upper edge it does not exceed.
 */
final class Binner {

    final int bins;
    final float[] edges;
    final int edgeCount;
    final int features;
    private final Tensor edgeTensor;

    Binner(Tensor x, int maxBins) {
        features = x.dim(1);
        int n = x.dim(0);
        float[] xs = x.toFloatArray();
        edgeCount = maxBins - 1;
        bins = maxBins;
        edges = new float[features * edgeCount];
        Arrays.fill(edges, Float.MAX_VALUE);
        float[] col = new float[n];
        for (int f = 0; f < features; f++) {
            for (int i = 0; i < n; i++) {
                col[i] = xs[i * features + f];
            }
            Arrays.sort(col);
            int distinct = 1;
            for (int i = 1; i < n; i++) {
                if (col[i] != col[i - 1]) {
                    distinct++;
                }
            }
            float[] u = new float[distinct];
            u[0] = col[0];
            for (int i = 1, k = 1; i < n; i++) {
                if (col[i] != col[i - 1]) {
                    u[k++] = col[i];
                }
            }
            int k = 0;
            if (distinct <= maxBins) {
                for (int i = 0; i + 1 < distinct; i++) {
                    edges[f * edgeCount + k++] = (u[i] + u[i + 1]) * 0.5f;
                }
            } else {
                float last = -Float.MAX_VALUE;
                for (int q = 1; q < maxBins; q++) {
                    double pos = (double) q / maxBins * (n - 1);
                    int lo = (int) Math.floor(pos);
                    int hi = Math.min(lo + 1, n - 1);
                    float v = (float) (col[lo] + (col[hi] - col[lo]) * (pos - lo));
                    if (v > last) {
                        edges[f * edgeCount + k++] = v;
                        last = v;
                    }
                }
            }
        }
        edgeTensor = Tensor.of(edges, features, edgeCount);
    }

    /** INT32 bins [n, F] of x. */
    Tensor bin(Tensor x) {
        if (x.dim(1) != features) {
            throw new IllegalArgumentException("fitted on " + features + " features, got " + x.shape());
        }
        return Trees.bin(x, edgeTensor);
    }
}
