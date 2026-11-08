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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.io.TempDir;

/** The GGUF reader on the fixture file, with operators on jitNN's host reference backend. */
class GgufFileTest {

    @Test
    void header() throws IOException {
        GgufFile f = GgufChecks.fixture();
        assertEquals(3, f.version());
        assertEquals(32, f.alignment());
        assertEquals(0, f.dataOffset() % 32);
        assertEquals("llama", f.architecture());
        assertEquals(30, f.tensorInfos().size());
    }

    @Test
    void metadataOfEveryType() throws IOException {
        GgufFile f = GgufChecks.fixture();
        assertEquals(250, f.integer("test.u8"));
        assertEquals(-100, f.integer("test.i8"));
        assertEquals(60000, f.integer("test.u16"));
        assertEquals(-30000, f.integer("test.i16"));
        assertEquals(3_000_000_000L, f.longValue("test.u32"));
        assertThrows(IllegalArgumentException.class, () -> f.integer("test.u32"));
        assertEquals(-2_000_000_000, f.integer("test.i32"));
        assertEquals(1.5f, f.floatValue("test.f32"));
        assertTrue(f.bool("test.bool"));
        assertEquals("héllo, wörld", f.string("test.string"));
        assertEquals((1L << 40) + 3, f.longValue("test.u64"));
        assertEquals(-(1L << 40), f.longValue("test.i64"));
        assertEquals(2.25f, f.floatValue("test.f64"));
        assertArrayEquals(new int[] { 1, -2, 3 }, f.ints("test.ints"));
        assertArrayEquals(new float[] { 0.5f, -1.25f }, f.floats("test.floats"));
        assertArrayEquals(new String[] { "a", "βγ", "" }, f.strings("test.strings"));
        assertArrayEquals(new boolean[] { true, false }, (boolean[]) f.get("test.bools"));
        Object[] nested = (Object[]) f.get("test.nested");
        assertEquals(2, nested.length);
        assertArrayEquals(new int[] { 1, 2 }, (int[]) nested[0]);
        assertArrayEquals(new int[] { 3 }, (int[]) nested[1]);
        assertArrayEquals(new long[] { 7, 4_000_000_000L }, f.longs("test.u32s"));
        assertArrayEquals(new double[] { 0.125, -3.5 }, (double[]) f.get("test.f64s"));
        assertEquals(2, f.integer("llama.block_count"));
        assertEquals(7, f.integer("llama.missing", 7));
        assertNull(f.get("llama.missing"));
        assertThrows(IllegalArgumentException.class, () -> f.string("llama.missing"));
        assertThrows(IllegalArgumentException.class, () -> f.string("test.u8"));
    }

    @Test
    void shapesInJitnnOrder() throws IOException {
        GgufFile f = GgufChecks.fixture();
        assertArrayEquals(new long[] { 3, 512 }, f.info("w.Q4_K").shape());
        assertEquals(GgmlType.Q4_K, f.info("w.Q4_K").type());
        assertEquals(3 * 512 / 256 * 144, f.info("w.Q4_K").bytes());
        assertArrayEquals(new long[] { 5 }, f.info("axes.1d").shape());
        assertArrayEquals(new long[] { 2, 3, 4 }, f.info("axes.3d").shape());
        Tensor t = f.tensor("axes.3d");
        assertEquals(List.of(2, 3, 4), List.of(t.dim(0), t.dim(1), t.dim(2)));
        float[] v = t.toFloatArray();
        for (int i = 0; i < 24; i++) {
            assertEquals(i, v[i]); // numpy's row-major order is jitNN's
        }
        assertArrayEquals(new float[] { 0, 1, 2, 3, 4 }, f.tensor("axes.1d").toFloatArray());
        Tensor ints = f.tensor("ints.I32");
        assertEquals(DType.I32, ints.dtype());
        assertArrayEquals(new int[] { 1, -2, 3, 4, 5, -6 }, ints.toIntArray());
    }

    @Test
    void typesWithoutJitnnCounterpart() throws IOException {
        GgufFile f = GgufChecks.fixture();
        assertEquals(GgmlType.Q8_K, f.info("w.Q8_K").type());
        assertThrows(UnsupportedOperationException.class, () -> f.tensor("w.Q8_K"));
        assertThrows(UnsupportedOperationException.class, f::tensors);
        assertEquals(3 * 512 / 256 * 292, f.raw("w.Q8_K").byteSize());
        assertThrows(IllegalArgumentException.class, () -> f.tensor("no.such.tensor"));
    }

    @Test
    void rejectsOtherFiles(@TempDir Path dir) throws IOException {
        Path notGguf = dir.resolve("a.bin");
        Files.write(notGguf, new byte[64]);
        assertThrows(IOException.class, () -> GgufFile.open(notGguf));
        byte[] whole = Files.readAllBytes(GgufChecks.fixture().path());
        Path truncated = dir.resolve("t.gguf");
        Files.write(truncated, java.util.Arrays.copyOf(whole, 600));
        assertThrows(IOException.class, () -> GgufFile.open(truncated));
    }

    /** A GGUF header with one F32 tensor of the given dimension and offset, and optional metadata bytes. */
    private static byte[] crafted(long dim, long offset, byte[] metadata, int kvCount) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(4096).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.putInt(0x46554747).putInt(3).putLong(1).putLong(kvCount);
        b.put(metadata);
        byte[] name = "w".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        b.putLong(name.length).put(name).putInt(1).putLong(dim).putInt(0).putLong(offset);
        return java.util.Arrays.copyOf(b.array(), b.position() + 64);
    }

    @Test
    void rejectsCraftedSizes(@TempDir Path dir) throws IOException {
        // a dimension of 2^64 - 1 (negative as a long) and an offset that overflows past the data start
        Path hugeDim = dir.resolve("dim.gguf");
        Files.write(hugeDim, crafted(-1L, 0, new byte[0], 0));
        assertThrows(IOException.class, () -> GgufFile.open(hugeDim));
        Path hugeOffset = dir.resolve("offset.gguf");
        Files.write(hugeOffset, crafted(4, Long.MAX_VALUE - 31, new byte[0], 0));
        assertThrows(IOException.class, () -> GgufFile.open(hugeOffset));
        // arrays nested 200 deep: rejected instead of overflowing the stack
        java.nio.ByteBuffer kv = java.nio.ByteBuffer.allocate(4096).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        byte[] key = "deep".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        kv.putLong(key.length).put(key).putInt(9);
        for (int i = 0; i < 200; i++) {
            kv.putInt(9).putLong(1);
        }
        kv.putInt(4).putLong(0);
        Path deep = dir.resolve("deep.gguf");
        Files.write(deep, crafted(4, 0, java.util.Arrays.copyOf(kv.array(), kv.position()), 1));
        assertThrows(IOException.class, () -> GgufFile.open(deep));
    }

    @Test
    void readOnlyFileIsCopied(@TempDir Path dir) throws IOException {
        Path ro = dir.resolve("ro.gguf");
        Files.copy(GgufChecks.fixture().path(), ro);
        assertTrue(ro.toFile().setWritable(false));
        GgufFile f = GgufFile.open(ro);
        GgufChecks.close("read-only Q4_K", f.tensor("expect.Q4_K").toFloatArray(), f.tensor("w.Q4_K").toFloatArray());
    }

    @Test
    void mappingLeavesTheFileUnchanged(@TempDir Path dir) throws IOException {
        Path rw = dir.resolve("rw.gguf");
        Files.copy(GgufChecks.fixture().path(), rw);
        byte[] before = Files.readAllBytes(rw);
        GgufFile f = GgufFile.open(rw);
        for (String name : f.tensorInfos().keySet()) {
            if (!name.equals("w.Q8_K")) {
                f.tensor(name).toFloatArray(); // writes each array header into its private mapping
            }
        }
        assertArrayEquals(before, Files.readAllBytes(rw));
    }

    @TestFactory
    List<DynamicTest> weightsOnReference() throws IOException {
        return GgufChecks.weights(Backend.REFERENCE);
    }
}
