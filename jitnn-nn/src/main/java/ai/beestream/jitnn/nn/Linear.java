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
 * y = x W^T + b with W of shape [out, in]. Initialised like PyTorch's {@code nn.Linear}: weight
 * and bias uniform in [-1/sqrt(in), 1/sqrt(in)).
 */
public final class Linear extends Module {

    private final Tensor weight;
    private final Tensor bias;

    /** A linear layer from {@code in} to {@code out} features with a bias, initialised from {@code seed}. */
    public Linear(int in, int out, long seed) {
        this(in, out, true, seed);
    }

    /**
     * A linear layer from {@code in} to {@code out} features, with or without a bias, initialised from
     * {@code seed}.
     */
    public Linear(int in, int out, boolean bias, long seed) {
        float bound = (float) (1.0 / Math.sqrt(in));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, out, in));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    /** The weight [out, in]. */
    public Tensor weight() {
        return weight;
    }

    /** The bias [out], or null without one. */
    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.linear(x, weight, bias);
    }

    /** relu(forward(x)) as one fused GEMM; {@link Sequential} uses it for a Linear followed by ReLU. */
    public Tensor forwardRelu(Tensor x) {
        return F.linearRelu(x, weight, bias);
    }
}
