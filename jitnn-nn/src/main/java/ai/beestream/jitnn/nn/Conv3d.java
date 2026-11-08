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
 * 3-D convolution over [N, C, ...] with a cubic kernel, optionally grouped, PyTorch's initialisation:
 * weight and bias uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in = in * kernel^3.
 */
public final class Conv3d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int stride;
    private final int padding;
    private final int dilation;
    private final int groups;

    /**
     * A 3-D convolution from {@code in} to {@code out} channels with a cubic kernel, initialised from
     * {@code seed}.
     */
    public Conv3d(int in, int out, int kernel, int stride, int padding, int dilation, boolean bias, long seed) {
        this(in, out, kernel, stride, padding, dilation, 1, bias, seed);
    }

    /** With {@code groups} groups of in / groups input channels each (fan_in = in / groups * kernel^3). */
    public Conv3d(int in, int out, int kernel, int stride, int padding, int dilation, int groups, boolean bias, long seed) {
        if (in % groups != 0 || out % groups != 0) {
            throw new IllegalArgumentException(in + " and " + out + " channels do not split into " + groups + " groups");
        }
        this.groups = groups;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        float bound = (float) (1.0 / Math.sqrt(in / groups * Math.pow(kernel, 3)));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, out, in / groups, kernel, kernel, kernel));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    /** The weight [out, in, kernel, kernel, kernel]. */
    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.conv3d(x, weight, bias, stride, padding, dilation, groups);
    }
}
