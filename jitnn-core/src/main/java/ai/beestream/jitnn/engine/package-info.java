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
/**
 * The execution engine, which records operators in a per-thread trace and, at a sync point, compiles
 * the trace into one TornadoVM task graph, cached by structure and replayed, or runs it on the host
 * with the double-precision reference. Also the device choice, run-time autotuning, and the switch
 * to vendor libraries.
 *
 * <p><b>Internal: not part of the API</b>, and may change in any release. Use
 * {@link ai.beestream.jitnn.JitnnRuntime} and {@link ai.beestream.jitnn.JitnnConfig}. The exceptions are
 * the types a {@link ai.beestream.jitnn.CustomOp} works with, {@link ai.beestream.jitnn.engine.Emitter}
 * and {@link ai.beestream.jitnn.engine.Node}, the only ones in the published API documentation.
 */
package ai.beestream.jitnn.engine;
