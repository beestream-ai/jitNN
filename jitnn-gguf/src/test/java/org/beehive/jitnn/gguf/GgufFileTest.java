package org.beehive.jitnn.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
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
        return GgufChecks.weights(Engine.Backend.REFERENCE);
    }
}
