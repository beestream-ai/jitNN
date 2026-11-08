/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

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

    /**
     * Attention over {@code dim} features in {@code heads} heads ({@code dim} divisible by it), with or
     * without projection biases, initialised from {@code seed}.
     */
    public MultiheadAttention(int dim, int heads, boolean bias, long seed) {
        if (dim % heads != 0) {
            throw new IllegalArgumentException(dim + " features do not split into " + heads + " heads");
        }
        this.heads = heads;
        this.qkv = child(new Linear(dim, 3 * dim, bias, seed));
        this.out = child(new Linear(dim, dim, bias, seed + 1));
    }

    /** The fused q/k/v input projection [3 dim, dim]. */
    public Linear qkv() {
        return qkv;
    }

    /** The output projection [dim, dim]. */
    public Linear outProjection() {
        return out;
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, null, false);
    }

    /**
     * Attention over x [B, T, D] with an optional additive {@code mask} (null for none) and optional causal
     * masking.
     */
    public Tensor forward(Tensor x, Tensor mask, boolean causal) {
        return F.multiHeadAttention(x, qkv.weight(), qkv.bias(), out.weight(), out.bias(), heads, mask, causal);
    }
}
