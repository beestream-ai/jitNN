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
package ai.beestream.jitnn.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.training.GradientCollector;

import ai.beestream.jitnn.Tensor;

/**
 * Back-propagation through jitNN's autograd. A new collector clears the gradients of every array
 * that tracks them, so each training step starts from zero, as DJL's trainers expect.
 */
final class JitnnGradientCollector implements GradientCollector {

    private final JitnnEngine engine;

    JitnnGradientCollector(JitnnEngine engine) {
        this.engine = engine;
        engine.zeroGradients();
    }

    @Override
    public void backward(NDArray target) {
        Tensor t = ((JitnnNDManager) target.getManager()).from(target).tensor();
        if (t.numel() == 1) {
            t.backward();
        } else {
            t.backward(Tensor.ones(t.shape().dims()));
        }
        engine.storeGradients();
    }

    @Override
    public void zeroGradients() {
        engine.zeroGradients();
    }

    @Override
    public void close() {
    }
}
