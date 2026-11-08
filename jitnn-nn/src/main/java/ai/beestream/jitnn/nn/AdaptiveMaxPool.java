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

/** Adaptive max pooling to fixed output sizes (1 to 3 spatial axes), PyTorch's nn.AdaptiveMaxPool1d/2d/3d. */
public final class AdaptiveMaxPool extends Module {

    private final int[] out;

    /** Pools to the output sizes {@code out}, one per spatial axis. */
    public AdaptiveMaxPool(int... out) {
        this.out = out.clone();
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.adaptiveMaxPool(x, out);
    }
}
