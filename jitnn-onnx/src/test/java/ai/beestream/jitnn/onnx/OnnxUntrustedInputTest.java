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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Model files are untrusted input: sizes, offsets, nesting and external-data paths that a crafted
 * file declares are checked before anything is allocated or read. Host only, no device.
 */
class OnnxUntrustedInputTest {

    /** A GraphProto with one FLOAT initializer [4] whose data is external at {@code location}. */
    private static ByteBuffer graphWithExternal(String location, long offset, long length) {
        ProtoWriter tensor = new ProtoWriter().varint(1, 4).varint(2, OnnxModel.FLOAT).string(8, "w")
                .message(13, new ProtoWriter().string(1, "location").string(2, location))
                .message(13, new ProtoWriter().string(1, "offset").string(2, Long.toString(offset)))
                .message(13, new ProtoWriter().string(1, "length").string(2, Long.toString(length)))
                .varint(14, 1);
        return ByteBuffer.wrap(new ProtoWriter().message(5, tensor).bytes());
    }

    @Test
    void externalDataStaysInTheModelDirectory(@TempDir Path root) throws IOException {
        Path model = Files.createDirectories(root.resolve("model"));
        Files.write(root.resolve("secret.bin"), new byte[16]);
        Files.write(model.resolve("w.bin"), new byte[16]);

        IOException outside = assertThrows(IOException.class, () -> OnnxModel.graph(new Proto(graphWithExternal("../secret.bin", 0, 16)), model, new HashMap<>()));
        assertTrue(outside.getMessage().contains("outside"), outside.getMessage());
        String absolute = root.resolve("secret.bin").toAbsolutePath().toString();
        assertThrows(IOException.class, () -> OnnxModel.graph(new Proto(graphWithExternal(absolute, 0, 16)), model, new HashMap<>()));

        // inside the directory, but past the end of the file
        assertThrows(IOException.class, () -> OnnxModel.graph(new Proto(graphWithExternal("w.bin", 8, 100)), model, new HashMap<>()));
        assertThrows(IOException.class, () -> OnnxModel.graph(new Proto(graphWithExternal("w.bin", -4, 8)), model, new HashMap<>()));
        // and a valid one still loads
        OnnxModel.graph(new Proto(graphWithExternal("w.bin", 0, 16)), model, new HashMap<>());
    }

    @Test
    void declaredSizesAreCheckedBeforeAllocating() {
        ByteBuffer four = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        // 2^30 elements declared over 4 bytes: rejected, not a 4 GB array
        OnnxModel.TensorData big = new OnnxModel.TensorData("big", new long[] { 1 << 20, 1 << 10 }, OnnxModel.FLOAT, four);
        assertThrows(IllegalArgumentException.class, big::floats);
        assertThrows(IllegalArgumentException.class, big::longs);
        assertThrows(IllegalArgumentException.class, () -> new OnnxModel.TensorData("neg", new long[] { -1, 4 }, OnnxModel.FLOAT, four).numel());
        assertThrows(IllegalArgumentException.class, () -> new OnnxModel.TensorData("wide", new long[] { 1L << 40, 1L << 40 }, OnnxModel.FLOAT, four).numel());
        assertEquals(1, new OnnxModel.TensorData("one", new long[] { 1 }, OnnxModel.FLOAT, four).floats().length);
    }

    @Test
    void malformedProtobufIsRejected() {
        // a length-delimited field claiming 1000 bytes in a 4-byte message
        Proto tooLong = new Proto(ByteBuffer.wrap(new byte[] { 0x0A, (byte) 0xE8, 0x07, 0x00 }));
        tooLong.next();
        assertThrows(IllegalArgumentException.class, tooLong::message);

        // messages nested 200 deep
        ProtoWriter w = new ProtoWriter();
        for (int i = 0; i < 200; i++) {
            w = new ProtoWriter().message(1, w);
        }
        Proto p = new Proto(ByteBuffer.wrap(w.bytes()));
        assertThrows(IllegalArgumentException.class, () -> {
            Proto q = p;
            while (q.next()) {
                q = q.message();
            }
        });
    }
}
