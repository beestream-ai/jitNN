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

/** Residual connection: relu(main(x) + shortcut(x)) or main(x) + shortcut(x). */
public final class Add extends Module {

    private final Module main;
    private final Module shortcut;
    private final boolean relu;

    /** {@code shortcut} null means identity. */
    public Add(Module main, Module shortcut, boolean relu) {
        this.main = child(main);
        this.shortcut = shortcut == null ? null : child(shortcut);
        this.relu = relu;
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor y = main.forward(x).add(shortcut == null ? x : shortcut.forward(x));
        return relu ? y.relu() : y;
    }
}
