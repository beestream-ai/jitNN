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
 * EmbeddingBag: bags of INT32 indices (one flat list plus bag offsets) to the sum, mean or max of
 * their embedding rows, PyTorch's nn.EmbeddingBag with standard normal initial weights.
 */
public final class EmbeddingBag extends Module {

    private final Tensor weight;
    private final F.BagMode mode;

    /**
     * A table of {@code count} vectors of size {@code dim}, standard normal from {@code seed}, whose bags
     * combine as {@code mode} says.
     */
    public EmbeddingBag(int count, int dim, F.BagMode mode, long seed) {
        this.weight = parameter(Tensor.normal(seed, 0.0f, 1.0f, count, dim));
        this.mode = mode;
    }

    /** The table [count, dim]. */
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
