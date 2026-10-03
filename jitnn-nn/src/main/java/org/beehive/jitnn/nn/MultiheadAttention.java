package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * Multi-head self-attention on [B, T, D] inputs, like PyTorch's {@code nn.MultiheadAttention}
 * with batch_first: one fused q/k/v projection [3D, D], {@code heads} heads of D / heads
 * features, and an output projection. Optional causal masking and an additive mask broadcast to
 * [B, heads, T, T] (for example [B, 1, 1, T] for padding).
 */
public final class MultiheadAttention extends Module {

    private final Linear qkv;
    private final Linear out;
    private final int heads;

    public MultiheadAttention(int dim, int heads, boolean bias, long seed) {
        if (dim % heads != 0) {
            throw new IllegalArgumentException(dim + " features do not split into " + heads + " heads");
        }
        this.heads = heads;
        this.qkv = child(new Linear(dim, 3 * dim, bias, seed));
        this.out = child(new Linear(dim, dim, bias, seed + 1));
    }

    public Linear qkv() {
        return qkv;
    }

    public Linear outProjection() {
        return out;
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, null, false);
    }

    public Tensor forward(Tensor x, Tensor mask, boolean causal) {
        return F.multiHeadAttention(x, qkv.weight(), qkv.bias(), out.weight(), out.bias(), heads, mask, causal);
    }
}
