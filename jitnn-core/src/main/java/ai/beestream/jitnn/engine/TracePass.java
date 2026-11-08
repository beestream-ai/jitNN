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
package ai.beestream.jitnn.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;

import ai.beestream.jitnn.Tensor;

/**
 * A rewrite of a recorded trace before it runs (fusion, layout changes). Passes are found with
 * {@link ServiceLoader} ({@code META-INF/services/ai.beestream.jitnn.engine.TracePass}) and run in
 * the order of {@link #order()}; the operators register theirs, so the engine does not depend on
 * them.
 */
public interface TracePass {

    /** The rewritten trace; {@code wanted} are the tensors the caller will read. */
    List<Node> rewrite(List<Node> nodes, Tensor[] wanted);

    /** Lower runs first. */
    default int order() {
        return 0;
    }

    /** Runs every registered pass over {@code nodes}. */
    static List<Node> applyAll(List<Node> nodes, Tensor[] wanted) {
        for (TracePass p : Registered.PASSES) {
            nodes = p.rewrite(nodes, wanted);
        }
        return nodes;
    }

    /** The registered passes, loaded once. */
    final class Registered {
        static final List<TracePass> PASSES = load();

        private Registered() {
        }

        private static List<TracePass> load() {
            List<TracePass> passes = new ArrayList<>();
            ServiceLoader.load(TracePass.class, TracePass.class.getClassLoader()).forEach(passes::add);
            passes.sort(java.util.Comparator.comparingInt(TracePass::order));
            return List.copyOf(passes);
        }
    }
}
