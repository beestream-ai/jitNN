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
 * Transposed 2-D convolution over NCHW with a square kernel, PyTorch's {@code nn.ConvTranspose2d}
 * semantics and initialisation: weight [in, out/groups, k, k] and bias uniform in
 * [-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in = out/groups * k * k.
 */
public final class ConvTranspose2d extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int stride;
    private final int padding;
    private final int outputPadding;
    private final int dilation;
    private final int groups;

    /**
     * A transposed convolution from {@code in} to {@code out} channels with a bias, no output padding or
     * dilation, one group.
     */
    public ConvTranspose2d(int in, int out, int kernel, int stride, int padding, long seed) {
        this(in, out, kernel, stride, padding, 0, 1, 1, true, seed);
    }

    /**
     * A transposed convolution from {@code in} to {@code out} channels in {@code groups} groups, initialised
     * from {@code seed}.
     */
    public ConvTranspose2d(int in, int out, int kernel, int stride, int padding, int outputPadding, int dilation, int groups, boolean bias, long seed) {
        this.stride = stride;
        this.padding = padding;
        this.outputPadding = outputPadding;
        this.dilation = dilation;
        this.groups = groups;
        float bound = (float) (1.0 / Math.sqrt((double) out / groups * kernel * kernel));
        this.weight = parameter(Tensor.uniform(seed, -bound, bound, in, out / groups, kernel, kernel));
        this.bias = bias ? parameter(Tensor.uniform(seed + 0x9E3779B97F4A7C15L, -bound, bound, out)) : null;
    }

    /** The weight [in, out/groups, kernel, kernel]. */
    public Tensor weight() {
        return weight;
    }

    /** The bias [out], or null without one. */
    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.convTranspose2d(x, weight, bias, stride, padding, outputPadding, dilation, groups);
    }
}
