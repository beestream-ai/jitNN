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
package ai.beestream.jitnn.gguf;

/**
 * One tensor listed in a GGUF file.
 *
 * @param name   the tensor's name, for example {@code blk.0.attn_q.weight}
 * @param shape  the shape in jitNN order, outermost axis first; GGUF lists axes innermost first, so
 *               a weight stored as {@code [in, out]} there is {@code [out, in]} here
 * @param typeId the ggml type id
 * @param type   the ggml type, or {@code null} for an id this reader does not know
 * @param offset the byte offset of the data from the start of the file
 * @param bytes  the size of the data in bytes, or -1 when the type is unknown
 */
public record GgufTensorInfo(String name, long[] shape, int typeId, GgmlType type, long offset, long bytes) {

    /** Number of values. */
    public long numel() {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        return n;
    }
}
