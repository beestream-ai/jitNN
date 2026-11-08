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
 * Optimisers with PyTorch's semantics (SGD, Adam and AdamW, AdaGrad, RMSProp, AdaDelta, AdaMax,
 * NAdam, LAMB, LARS), learning-rate schedules ({@link ai.beestream.jitnn.nn.optim.LRScheduler}) and dynamic
 * loss scaling for FP16 training ({@link ai.beestream.jitnn.nn.optim.GradScaler}).
 */
package ai.beestream.jitnn.nn.optim;
