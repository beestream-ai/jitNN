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
 * The GGUF reader (llama.cpp's model format): {@link ai.beestream.jitnn.gguf.GgufFile} gives a file's
 * metadata and its tensors, memory-mapped without a copy, in their quantised blocks or as F32, F16
 * and BF16.
 */
package ai.beestream.jitnn.gguf;
