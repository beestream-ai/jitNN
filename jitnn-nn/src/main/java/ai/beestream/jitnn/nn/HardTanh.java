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

import ai.beestream.jitnn.Tensor;

/** HardTanh: x clamped to [min, max]. */
public final class HardTanh extends Module {

    private final float min;
    private final float max;

    /** Clamps to [min, max] (PyTorch's default -1, 1). */
    public HardTanh(float min, float max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public Tensor forward(Tensor x) {
        return x.hardTanh(min, max);
    }
}
