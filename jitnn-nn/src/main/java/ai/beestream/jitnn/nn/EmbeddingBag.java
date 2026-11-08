package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * EmbeddingBag: bags of INT32 indices (one flat list plus bag offsets) to the sum, mean or max of
 * their embedding rows, PyTorch's nn.EmbeddingBag with standard normal initial weights.
 */
public final class EmbeddingBag extends Module {

    private final Tensor weight;
    private final F.BagMode mode;

    public EmbeddingBag(int count, int dim, F.BagMode mode, long seed) {
        this.weight = parameter(Tensor.normal(seed, 0.0f, 1.0f, count, dim));
        this.mode = mode;
    }

    public Tensor weight() {
        return weight;
    }

    /** One row per bag: bag b covers indices offsets[b] .. offsets[b + 1] - 1. */
    public Tensor forward(Tensor indices, int[] offsets) {
        return F.embeddingBag(indices, offsets, weight, mode);
    }

    /** Every index its own bag. */
    @Override
    public Tensor forward(Tensor indices) {
        int[] offsets = new int[indices.numel()];
        for (int i = 0; i < offsets.length; i++) {
            offsets[i] = i;
        }
        return forward(indices, offsets);
    }
}
