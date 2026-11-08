package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Module;

/**
 * GloVe (Pennington et al., 2014): word and context vectors with biases fitted to the logarithm
 * of the co-occurrence counts, weighted by f(x) = min(1, (x / xMax)^alpha). The paper trains it
 * with AdaGrad; the word vectors are the sum of the two tables.
 */
public final class GloVe extends Module {

    private final Tensor w;
    private final Tensor wt;
    private final Tensor b;
    private final Tensor bt;
    private final int dim;
    private final float xMax;
    private final float alpha;

    public GloVe(int vocab, int dim, long seed) {
        this(vocab, dim, 100.0f, 0.75f, seed);
    }

    public GloVe(int vocab, int dim, float xMax, float alpha, long seed) {
        this.dim = dim;
        this.xMax = xMax;
        this.alpha = alpha;
        this.w = parameter(Tensor.uniform(seed, -0.5f / dim, 0.5f / dim, vocab, dim));
        this.wt = parameter(Tensor.uniform(seed + 1, -0.5f / dim, 0.5f / dim, vocab, dim));
        this.b = parameter(Tensor.zeros(vocab));
        this.bt = parameter(Tensor.zeros(vocab));
    }

    /** The loss of a batch of co-occurrences x at (rows, cols), all > 0. */
    public Tensor loss(int[] rows, int[] cols, float[] x) {
        int n = rows.length;
        float[] logX = new float[n];
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) {
            logX[i] = (float) Math.log(x[i]);
            weight[i] = (float) Math.min(1.0, Math.pow(x[i] / xMax, alpha));
        }
        Tensor ri = Tensor.of(rows, n);
        Tensor ci = Tensor.of(cols, n);
        return F.gloveLoss(F.embedding(ri, w), F.embedding(ci, wt), F.embedding(ri, b.unsqueeze(1)).reshape(n), F.embedding(ci, bt.unsqueeze(1)).reshape(n),
                Tensor.of(logX, n), Tensor.of(weight, n));
    }

    /** The word vectors W + W~ of ids x. */
    @Override
    public Tensor forward(Tensor x) {
        return F.embedding(x, w).add(F.embedding(x, wt));
    }

    /** All word vectors [vocab, dim] on the host. */
    public float[] vectors() {
        return w.add(wt).toFloatArray();
    }

    public int dim() {
        return dim;
    }
}
