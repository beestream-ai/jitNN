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

import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * Inverted dropout in training mode, identity in evaluation mode. Each forward pass draws a new
 * seed on the host, so a replayed plan gets a new mask without recompiling; each Dropout module
 * has its own Philox stream.
 */
public final class Dropout extends Module {

    private static final AtomicInteger STREAMS = new AtomicInteger(1);

    private final float p;
    private final int stream = STREAMS.getAndIncrement();
    private final SplittableRandom seeds;

    /**
     * Dropout that zeroes each element with probability {@code p} and scales the rest by 1 / (1 - p); masks
     * are seeded from {@code seed}.
     */
    public Dropout(float p, long seed) {
        this.p = p;
        this.seeds = new SplittableRandom(seed);
    }

    @Override
    public Tensor forward(Tensor x) {
        if (!isTraining() || p == 0) {
            return x;
        }
        return F.dropout(x, p, Tensor.of(new int[] { seeds.nextInt(), seeds.nextInt() }, 2), stream);
    }
}
