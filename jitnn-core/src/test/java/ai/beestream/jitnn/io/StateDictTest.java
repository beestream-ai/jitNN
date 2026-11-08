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
package ai.beestream.jitnn.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

class StateDictTest {

    @TempDir
    Path dir;

    @BeforeAll
    static void reference() {
        JitnnRuntime.setBackend(Backend.REFERENCE);
    }

    @Test
    void roundTripEveryType() {
        Tensor f32 = Tensor.persistent(Tensor.of(new float[] { 1.5f, -2, 3.25f, 0, 7, -0.125f }, 2, 3));
        Tensor f16 = Tensor.persistentHalf(new float[] { 0.5f, -1, 65504 }, 3);
        Tensor bf16 = Tensor.persistent(ShortArray.fromElements((short) 0x3f80, (short) 0xc000), DType.BF16, 2);
        Tensor i32 = Tensor.persistent(IntArray.fromElements(7, -3, 1 << 30, 0), DType.I32, 2, 2);
        Tensor scalar = Tensor.persistent(Tensor.of(new float[] { 42 }));
        Path file = dir.resolve("all.safetensors");
        new StateDict().put("a.f32", f32).put("a.f16", f16).put("b.bf16", bf16).put("b.i32", i32).put("s", scalar)
                .put("note", "quote \" backslash \\ newline \n").put("step", 12L).save(file);

        StateDict in = StateDict.load(file);
        assertEquals(List.of("a.f32", "a.f16", "b.bf16", "b.i32", "s"), List.copyOf(in.tensorNames()));
        assertEquals("quote \" backslash \\ newline \n", in.value("note"));
        assertEquals(12, in.intValue("step"));
        assertArrayEquals(f32.toFloatArray(), in.tensor("a.f32").toFloatArray());
        assertArrayEquals(f16.toFloatArray(), in.tensor("a.f16").toFloatArray());
        assertArrayEquals(new float[] { 1, -2 }, in.tensor("b.bf16").toFloatArray());
        assertArrayEquals(i32.toIntArray(), in.tensor("b.i32").toIntArray());
        assertArrayEquals(new int[0], in.shape("s"));
        assertArrayEquals(new float[] { 42 }, in.tensor("s").toFloatArray());
        assertArrayEquals(new int[] { 2, 3 }, in.sub("a").shape("f32"));

        Tensor target = Tensor.persistent(Tensor.zeros(2, 3));
        in.copyInto("a.f32", target);
        assertArrayEquals(f32.toFloatArray(), target.toFloatArray());
        assertThrows(IllegalArgumentException.class, () -> in.copyInto("a.f32", Tensor.persistent(Tensor.zeros(3, 2))));
        assertThrows(IllegalArgumentException.class, () -> in.copyInto("a.f32", Tensor.zeros(2, 3)));
    }

    @Test
    void dataIsAligned() throws IOException {
        Path file = dir.resolve("x.safetensors");
        new StateDict().put("odd", Tensor.persistent(Tensor.of(new float[] { 1, 2, 3 }, 3))).put("k", "v").save(file);
        long n = ByteBuffer.wrap(Files.readAllBytes(file), 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        assertEquals(0, (8 + n) % 8);
        assertEquals(8 + n + 12, Files.size(file));
    }

    /** A header as other writers produce it: spaces, entries out of offset order, metadata first or last. */
    @Test
    void readsForeignHeader() throws IOException {
        String json = "{ \"b\" : {\"dtype\":\"F32\", \"shape\":[1], \"data_offsets\":[8,12]},\n \"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}, \"__metadata__\":{\"format\":\"pt\"} }";
        byte[] header = json.getBytes(StandardCharsets.UTF_8);
        ByteBuffer b = ByteBuffer.allocate(8 + header.length + 12).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(header.length).put(header).putFloat(1).putFloat(2).putFloat(3);
        Path file = dir.resolve("foreign.safetensors");
        Files.write(file, b.array());
        StateDict in = StateDict.load(file);
        assertEquals(List.of("a", "b"), List.copyOf(in.tensorNames()));
        assertEquals("pt", in.value("format"));
        assertArrayEquals(new float[] { 1, 2 }, in.tensor("a").toFloatArray());
        assertArrayEquals(new float[] { 3 }, in.tensor("b").toFloatArray());
    }

    @Test
    void rejectsBadFiles() throws IOException {
        Path file = dir.resolve("bad.safetensors");
        Files.write(file, new byte[] { 1, 2, 3 });
        assertThrows(RuntimeException.class, () -> StateDict.load(file));
        assertThrows(IllegalArgumentException.class, () -> new StateDict().put("q", Tensor.quantized(DType.Q8_0, new byte[34], 32)));
        assertThrows(IllegalArgumentException.class, () -> new StateDict().put("x", "1").put("x", "2"));
    }
}
