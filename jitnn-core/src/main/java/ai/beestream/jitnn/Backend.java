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
package ai.beestream.jitnn;

/**
 * Where operators run.
 *
 * <p>The default is {@link #TORNADO} when a TornadoVM runtime is on the class path, else
 * {@link #CPU} when the JVM has the Vector API module, else {@link #REFERENCE};
 * {@code -Djitnn.backend} ({@code tornado}, {@code cpu} or {@code reference}) overrides it, and
 * {@link JitnnRuntime#setBackend} and {@link JitnnRuntime#with} change it per thread.
 */
public enum Backend {
    /**
     * Operators are recorded lazily; a sync point compiles everything recorded since the last one
     * into one TornadoVM task graph, caches the compiled plan by the trace's structure, and replays
     * it when the same structure comes back (a training step).
     */
    TORNADO,
    /**
     * Operators are recorded lazily as on {@link #TORNADO}; a sync point rewrites the trace with
     * the same fusion pass and runs it on the host, each operator with its multithreaded, vectorised
     * CPU kernel (Java Vector API) or, when it has none, its reference implementation. It needs no
     * TornadoVM runtime, but the JVM needs {@code --add-modules jdk.incubator.vector}.
     */
    CPU,
    /**
     * Operators run eagerly on the host with the Java reference implementations. Tests use it as
     * the oracle; it needs no TornadoVM runtime.
     */
    REFERENCE
}
