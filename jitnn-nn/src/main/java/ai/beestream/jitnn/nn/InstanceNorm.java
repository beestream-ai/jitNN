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
 * InstanceNorm over [N, C, ...]: each sample and channel normalised over its positions, PyTorch's
 * nn.InstanceNorm1d/2d/3d without running statistics; affine optional (PyTorch's default: none).
 */
public final class InstanceNorm extends Module {

    private final float eps;
    private final Tensor weight;
    private final Tensor bias;

    /**
     * InstanceNorm of {@code channels} channels with the given eps, with or without a per-channel affine
     * (weight 1, bias 0 at the start).
     */
    public InstanceNorm(int channels, float eps, boolean affine) {
        this.eps = eps;
        this.weight = affine ? parameter(Tensor.ones(channels)) : null;
        this.bias = affine ? parameter(Tensor.zeros(channels)) : null;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.instanceNorm(x, weight, bias, eps);
    }
}
