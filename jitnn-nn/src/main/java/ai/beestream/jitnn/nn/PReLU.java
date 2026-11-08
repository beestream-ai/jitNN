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

/** PReLU with a learnable slope, one shared or one per channel, initialised to {@code init} (PyTorch: 0.25). */
public final class PReLU extends Module {

    private final Tensor weight;

    /**
     * PReLU with {@code parameters} slopes (1 shared, or one per channel), each {@code init} at the start.
     */
    public PReLU(int parameters, float init) {
        this.weight = parameter(Tensor.full(init, parameters));
    }

    /** The slopes [parameters]. */
    public Tensor weight() {
        return weight;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.prelu(x, weight);
    }
}
