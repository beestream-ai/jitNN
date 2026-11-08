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
 * The calls into vendor libraries (cuBLAS, cuDNN, cuFFT, cuSPARSE on CUDA; MLX on Metal) that
 * operators make when {@code -Djitnn.native} enables them (see {@link ai.beestream.jitnn.engine.Native}).
 * {@link ai.beestream.jitnn.natives.Libs} is the only class operators use; it loads the
 * library-specific classes only once a library is known to be available.
 *
 * <p><b>Internal: not part of the API</b>, and may change in any release.
 */
package ai.beestream.jitnn.natives;
