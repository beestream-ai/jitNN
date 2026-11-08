package ai.beestream.jitnn.ml;

import ai.beestream.jitnn.Linalg;
import ai.beestream.jitnn.Tensor;

/**
 * Principal component analysis from the eigendecomposition of the covariance matrix, after
 * scikit-learn's PCA: components as rows, ordered by explained variance, each signed so that its
 * largest-magnitude entry is positive (scikit-learn's svd_flip convention).
 */
public final class Pca {

    private final int components;
    private Tensor mean;
    private Tensor basis;
    private float[] variance;
    private float[] ratio;

    public Pca(int components) {
        this.components = components;
    }

    public Pca fit(Tensor x) {
        Ml.check(x, "PCA");
        int n = x.dim(0);
        int d = x.dim(1);
        if (components > d) {
            throw new IllegalArgumentException("PCA: " + components + " components of " + d + " features");
        }
        mean = x.mean(0, true);
        Tensor xc = x.sub(mean);
        Tensor cov = xc.matmul(xc, true, false).mul(1.0f / (n - 1));
        Tensor[] eig = Linalg.eigh(cov);
        int[] order = new int[components];
        for (int i = 0; i < components; i++) {
            order[i] = d - 1 - i;
        }
        Tensor idx = Tensor.of(order, components);
        basis = eig[1].indexSelect(1, idx).transpose(0, 1).mul(1.0f);
        float[] all = eig[0].toFloatArray();
        double total = 0;
        for (float v : all) {
            total += Math.max(v, 0);
        }
        variance = new float[components];
        ratio = new float[components];
        for (int i = 0; i < components; i++) {
            variance[i] = Math.max(all[d - 1 - i], 0);
            ratio[i] = (float) (variance[i] / total);
        }
        return this;
    }

    /** The components [k, d]. */
    public Tensor components() {
        return basis;
    }

    public float[] explainedVariance() {
        return variance.clone();
    }

    public float[] explainedVarianceRatio() {
        return ratio.clone();
    }

    /** Projection [n, k] of x onto the components. */
    public Tensor transform(Tensor x) {
        return x.sub(mean).matmul(basis, false, true);
    }

    /** Back from the projection to the feature space. */
    public Tensor inverseTransform(Tensor z) {
        return z.matmul(basis).add(mean);
    }
}
