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
package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;

final class Checks {

    private Checks() {
    }

    static void f32(Tensor... ts) {
        for (Tensor t : ts) {
            // in half-precision inference FP16 inputs are converted when the operator is recorded
            if (t.dtype() != DType.F32 && !(t.dtype() == DType.F16 && ai.beestream.jitnn.Autocast.halfActivations())) {
                throw new IllegalArgumentException("expected an F32 tensor, got " + t);
            }
        }
    }

    static void rank(Tensor t, int rank) {
        if (t.rank() != rank) {
            throw new IllegalArgumentException("expected rank " + rank + ", got " + t);
        }
    }
}
