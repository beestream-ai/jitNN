package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Histogram-based decision trees (as in LightGBM and scikit-learn's HistGradientBoosting): features
 * quantised to bins, per-node histograms of per-sample statistics, routing of samples to children,
 * and prediction by walking trees. Split selection runs on the host from the histograms.
 */
public final class TreeKernels {

    private TreeKernels() {
    }

    /** bins[s, f] = the number of edges[f, :] below x[s, f] (edges [F, E] ascending): bin 0..E. */
    public static void bin(KernelContext kc, FloatArray x, FloatArray edges, IntArray bins, int features, int edgeCount, int total) {
        int id = kc.globalIdx;
        if (id < total) {
            int f = id % features;
            float v = x.get(id);
            int lo = 0;
            int hi = edgeCount;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (edges.get(f * edgeCount + mid) < v) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            bins.set(id, lo);
        }
    }

    /**
     * hist[node, f, b, c] = sum of stats[s, c] over samples s in node (node[s]) with bins[s, f] = b;
     * one thread per histogram cell, scanning the samples.
     */
    public static void histogram(KernelContext kc, IntArray bins, IntArray node, FloatArray stats, FloatArray hist, int samples, int features, int binCount, int channels,
            int total) {
        int id = kc.globalIdx;
        if (id < total) {
            int c = id % channels;
            int b = (id / channels) % binCount;
            int f = (id / (channels * binCount)) % features;
            int nd = id / (channels * binCount * features);
            float acc = 0.0f;
            for (int s = 0; s < samples; s++) {
                if ((node.get(s) == nd) & (bins.get(s * features + f) == b)) {
                    acc += stats.get(s * channels + c);
                }
            }
            hist.set(id, acc);
        }
    }

    /**
     * Moves each active sample to its child: out[s] = child[node] + (bins[s, feature[node]] > split[node]),
     * or -1 when the node is a leaf (feature -1) or the sample is inactive.
     */
    public static void route(KernelContext kc, IntArray bins, IntArray node, IntArray feature, IntArray split, IntArray child, IntArray out, int features, int samples) {
        int s = kc.globalIdx;
        if (s < samples) {
            int nd = node.get(s);
            int r = -1;
            if (nd >= 0) {
                int f = feature.get(nd);
                if (f >= 0) {
                    r = child.get(nd);
                    if (bins.get(s * features + f) > split.get(nd)) {
                        r = r + 1;
                    }
                }
            }
            out.set(s, r);
        }
    }

    /**
     * out[s, o] = sum over trees t of value[t, leaf, o], the leaf reached by sample s in tree t
     * (nodes [T, M]: feature -1 marks a leaf; left child at child[t, n], right at child + 1).
     */
    public static void predict(KernelContext kc, IntArray bins, IntArray feature, IntArray split, IntArray child, FloatArray value, FloatArray out, int features, int trees,
            int maxNodes, int outputs, int total) {
        int id = kc.globalIdx;
        if (id < total) {
            int s = id / outputs;
            int o = id % outputs;
            float acc = 0.0f;
            for (int t = 0; t < trees; t++) {
                int base = t * maxNodes;
                int nd = 0;
                int f = feature.get(base);
                while (f >= 0) {
                    nd = child.get(base + nd) + (bins.get(s * features + f) > split.get(base + nd) ? 1 : 0);
                    f = feature.get(base + nd);
                }
                acc += value.get((base + nd) * outputs + o);
            }
            out.set(id, acc);
        }
    }
}
