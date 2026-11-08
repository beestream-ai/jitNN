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
package ai.beestream.jitnn.onnx;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/** Protocol-buffer encoding: the few wire forms the ONNX messages use (varints and length-delimited fields). */
final class ProtoWriter {

    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    byte[] bytes() {
        return out.toByteArray();
    }

    int size() {
        return out.size();
    }

    ProtoWriter varint(int field, long value) {
        tag(field, 0);
        raw(value);
        return this;
    }

    ProtoWriter fixed32(int field, float value) {
        tag(field, 5);
        int b = Float.floatToRawIntBits(value);
        for (int i = 0; i < 4; i++) {
            out.write((b >>> (8 * i)) & 0xFF);
        }
        return this;
    }

    ProtoWriter string(int field, String s) {
        return bytes(field, s.getBytes(StandardCharsets.UTF_8));
    }

    ProtoWriter bytes(int field, byte[] b) {
        tag(field, 2);
        raw(b.length);
        out.writeBytes(b);
        return this;
    }

    ProtoWriter message(int field, ProtoWriter m) {
        return bytes(field, m.bytes());
    }

    private void tag(int field, int wire) {
        raw(((long) field << 3) | wire);
    }

    private void raw(long v) {
        while ((v & ~0x7FL) != 0) {
            out.write((int) ((v & 0x7F) | 0x80));
            v >>>= 7;
        }
        out.write((int) v);
    }
}
