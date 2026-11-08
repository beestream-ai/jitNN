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
 * The operators: each {@link ai.beestream.jitnn.engine.Op} here has a device implementation (one or
 * more kernels from {@link ai.beestream.jitnn.kernels}, or a vendor-library call), a host reference
 * and a gradient. {@link ai.beestream.jitnn.Tensor} and {@link ai.beestream.jitnn.F} are the public way to
 * call them; {@link ai.beestream.jitnn.ops.Fusion} rewrites traces to fuse them.
 *
 * <p><b>Internal: not part of the API</b>, and may change in any release.
 */
package ai.beestream.jitnn.ops;
