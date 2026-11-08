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
package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.Tensor;

/** Initialisers that another framework can reproduce bit for bit (Philox uniforms, scripts/pytorch/jitnn_init.py). */
final class Init {

    private Init() {
    }

    /** Uniform with mean 0 and standard deviation {@code std}: U(-std * sqrt(3), std * sqrt(3)). */
    static Tensor uniformStd(long seed, float std, int... dims) {
        float a = (float) (std * Math.sqrt(3.0));
        return Tensor.uniform(seed, -a, a, dims);
    }
}
