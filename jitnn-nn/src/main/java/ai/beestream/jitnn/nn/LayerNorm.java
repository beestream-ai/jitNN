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

/** LayerNorm over the last axis, with weight 1 and bias 0 initially, like PyTorch's {@code nn.LayerNorm}. */
public final class LayerNorm extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final float eps;

    /** LayerNorm over {@code features} values, with eps 1e-5 and a bias. */
    public LayerNorm(int features) {
        this(features, 1e-5f, true);
    }

    /** LayerNorm over {@code features} values with the given eps, with or without a bias. */
    public LayerNorm(int features, float eps, boolean bias) {
        this.weight = parameter(Tensor.ones(features));
        this.bias = bias ? parameter(Tensor.zeros(features)) : null;
        this.eps = eps;
    }

    /** The scale [features]. */
    public Tensor weight() {
        return weight;
    }

    /** The shift [features], or null without one. */
    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.layerNorm(x, weight, bias, eps);
    }
}
