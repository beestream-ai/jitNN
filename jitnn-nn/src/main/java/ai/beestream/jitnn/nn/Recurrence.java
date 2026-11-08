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

/** Parameter set-up shared by {@link LSTM} and {@link GRU}. */
final class Recurrence {

    private Recurrence() {
    }

    /**
     * Per layer and direction (forward, then reverse: PyTorch's order) {W_ih, W_hh, b_ih, b_hh},
     * uniform in [-1/sqrt(H), 1/sqrt(H)); seeds seed + 4 (layer * directions + direction) + i.
     * Layers after the first read the directions' outputs side by side (directions * H features).
     */
    static Tensor[][] init(Module owner, int input, int hidden, int layers, int directions, int gates, long seed) {
        float bound = (float) (1.0 / Math.sqrt(hidden));
        Tensor[][] w = new Tensor[layers * directions][];
        for (int l = 0; l < layers; l++) {
            int in = l == 0 ? input : hidden * directions;
            for (int d = 0; d < directions; d++) {
                int k = l * directions + d;
                long s = seed + 4L * k;
                w[k] = new Tensor[] {
                    owner.parameter(Tensor.uniform(s, -bound, bound, gates * hidden, in)),
                    owner.parameter(Tensor.uniform(s + 1, -bound, bound, gates * hidden, hidden)),
                    owner.parameter(Tensor.uniform(s + 2, -bound, bound, gates * hidden)),
                    owner.parameter(Tensor.uniform(s + 3, -bound, bound, gates * hidden)) };
            }
        }
        return w;
    }
}
