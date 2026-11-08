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
 * The jitNN API: {@link ai.beestream.jitnn.Tensor} (n-dimensional arrays, lazy on TornadoVM
 * devices), reverse-mode differentiation ({@link ai.beestream.jitnn.Tensor#backward()}, {@link ai.beestream.jitnn.Grad}),
 * the functional operators of {@link ai.beestream.jitnn.F}, linear algebra ({@link ai.beestream.jitnn.Linalg}),
 * Fourier transforms ({@link ai.beestream.jitnn.Fft}), sparse matrices ({@link ai.beestream.jitnn.SparseMatrix}),
 * mixed precision ({@link ai.beestream.jitnn.Autocast}) and user-defined operators
 * ({@link ai.beestream.jitnn.CustomOp}).
 */
package ai.beestream.jitnn;
