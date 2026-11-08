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

/** Nearest or bilinear upsampling of NCHW by a scale factor, PyTorch's {@code nn.Upsample}. */
public final class Upsample extends Module {

    private final float scale;
    private final boolean bilinear;
    private final boolean alignCorners;

    /**
     * Upsampling by {@code scale}, nearest or bilinear; {@code alignCorners} as in PyTorch (bilinear only).
     */
    public Upsample(float scale, boolean bilinear, boolean alignCorners) {
        this.scale = scale;
        this.bilinear = bilinear;
        this.alignCorners = alignCorners;
    }

    @Override
    public Tensor forward(Tensor x) {
        return bilinear ? F.upsampleBilinear(x, scale, alignCorners) : F.upsampleNearest(x, scale);
    }
}
