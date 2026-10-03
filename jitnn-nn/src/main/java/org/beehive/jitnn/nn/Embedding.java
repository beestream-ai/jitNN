package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * A lookup table of {@code count} vectors of size {@code dim}, applied to INT32 indices.
 * Initialised like PyTorch's {@code nn.Embedding}: standard normal entries.
 */
public final class Embedding extends Module {

    private final Tensor weight;
    private final int paddingIdx;

    public Embedding(int count, int dim, long seed) {
        this(count, dim, -1, seed);
    }

    /** Row {@code paddingIdx} receives no gradient (pass -1 for none). */
    public Embedding(int count, int dim, int paddingIdx, long seed) {
        this(Tensor.normal(seed, 0.0f, 1.0f, count, dim), paddingIdx);
    }

    /** An embedding with the given initial table [count, dim]. */
    public Embedding(Tensor init, int paddingIdx) {
        this.weight = parameter(init);
        this.paddingIdx = paddingIdx;
    }

    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor index) {
        return F.embedding(index, weight, paddingIdx);
    }
}
