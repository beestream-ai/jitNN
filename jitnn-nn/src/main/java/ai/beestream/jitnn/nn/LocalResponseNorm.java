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

/** Local response normalisation across channels, PyTorch's nn.LocalResponseNorm. */
public final class LocalResponseNorm extends Module {

    private final int size;
    private final float alpha;
    private final float beta;
    private final float k;

    /** Normalisation over {@code size} neighbouring channels: x / (k + alpha / size * sum x^2)^beta. */
    public LocalResponseNorm(int size, float alpha, float beta, float k) {
        this.size = size;
        this.alpha = alpha;
        this.beta = beta;
        this.k = k;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.localResponseNorm(x, size, alpha, beta, k);
    }
}
