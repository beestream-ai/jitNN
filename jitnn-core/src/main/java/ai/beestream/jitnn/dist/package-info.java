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
 * Several devices: collectives (all-reduce, broadcast, reduce-scatter, all-gather) among ranks,
 * staged through the host ({@link ai.beestream.jitnn.dist.HostCollective}) or run by NCCL on NVIDIA GPUs
 * ({@link ai.beestream.jitnn.dist.NcclCollective}), and {@link ai.beestream.jitnn.dist.ProcessGroup}, which
 * runs one rank per device.
 */
package ai.beestream.jitnn.dist;
