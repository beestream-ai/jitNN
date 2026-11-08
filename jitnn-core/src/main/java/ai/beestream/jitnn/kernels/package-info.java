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
 * The device kernels behind the operators, written in Java with TornadoVM's {@code KernelContext}
 * API so that each runs on CUDA, OpenCL and Metal: element-wise, reductions, GEMM (and tensor-core
 * GEMMs on NVIDIA), convolutions, attention, normalisation, FFT, linear algebra, sparse, quantised
 * and FP8 kernels. Operators in {@link ai.beestream.jitnn.ops} choose and launch them.
 */
package ai.beestream.jitnn.kernels;
