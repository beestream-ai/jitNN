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

/** RMSNorm over the last axis, with weight 1 initially, like PyTorch's {@code nn.RMSNorm}. */
public final class RMSNorm extends Module {

    private final Tensor weight;
    private final float eps;

    /** RMSNorm over {@code features} values with the given eps. */
    public RMSNorm(int features, float eps) {
        this.weight = parameter(Tensor.ones(features));
        this.eps = eps;
    }

    /** The scale [features]. */
    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.rmsNorm(x, weight, eps);
    }
}
