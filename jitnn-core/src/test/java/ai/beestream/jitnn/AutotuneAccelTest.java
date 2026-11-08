package ai.beestream.jitnn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;
import java.util.function.Supplier;

import ai.beestream.jitnn.engine.Autotune;
import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Run-time autotuning on the test device: a GEMM, a 3x3 convolution and a layer norm are measured
 * on first use, every variant must reproduce the host reference, the decision is saved and read
 * back, and the operators give the same results with it.
 */
class AutotuneAccelTest {

    @TempDir
    static Path dir;

    @AfterEach
    void off() {
        Autotune.setMode("off");
        Autotune.clear();
    }

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static float[] random(int n, long seed) {
        java.util.Random r = new java.util.Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.nextFloat() * 2 - 1;
        }
        return v;
    }

    /** Device and host results of {@code body}: the device runs with tuning on, twice (before and after the decision). */
    private static void check(String family, Supplier<Tensor> body, float tolerance) {
        Autotune.setDirectory(dir);
        Autotune.setMode("on");
        float[] first = body.get().toFloatArray();
        Map<String, String> decided = Autotune.decisions();
        String key = decided.keySet().stream().filter(k -> k.startsWith(family + ":")).findFirst().orElse(null);
        assertNotNull(key, "no " + family + " decision among " + decided);
        float[] second = body.get().toFloatArray();
        float[] reference = Engine.with(Engine.Backend.REFERENCE, () -> body.get().toFloatArray());
        float scale = 1e-6f;
        for (float v : reference) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < reference.length; i++) {
            assertEquals(reference[i], first[i], tolerance * scale, family + " before the decision, element " + i);
            assertEquals(reference[i], second[i], tolerance * scale, family + " with " + decided.get(key) + ", element " + i);
        }
        System.out.printf("[autotune] %s -> %s%n", key, decided.get(key));
    }

    @Test
    void gemm() throws Exception {
        float[] a = random(200 * 152, 1);
        float[] b = random(152 * 168, 2);
        check("gemm", () -> Tensor.of(a, 200, 152).matmul(Tensor.of(b, 152, 168)), 1e-4f);
        // saved, and read back by a run that only applies decisions
        Path file = Autotune.file();
        assertTrue(Files.exists(file), "no decisions file " + file);
        Properties p = new Properties();
        try (var in = Files.newInputStream(file)) {
            p.load(in);
        }
        assertTrue(p.stringPropertyNames().stream().anyMatch(k -> k.startsWith("gemm:200x168x152")), "saved: " + p);
        Autotune.clear();
        Autotune.setMode("use");
        assertTrue(Autotune.decisions().keySet().stream().anyMatch(k -> k.startsWith("gemm:200x168x152")));
    }

    @Test
    void convolution3x3() {
        float[] x = random(16 * 32 * 14 * 14, 3);
        float[] w = random(32 * 32 * 9, 4);
        check("conv3x3", () -> F.conv2d(Tensor.of(x, 16, 32, 14, 14), Tensor.of(w, 32, 32, 3, 3), null, 1, 1, 1, 1), 1e-3f);
    }

    @Test
    void layerNorm() {
        assumeCuda();
        float[] x = random(2048 * 512, 5);
        check("rownorm", () -> F.layerNorm(Tensor.of(x, 2048, 512), Tensor.ones(512), Tensor.zeros(512), 1e-5f), 1e-4f);
    }

    private static void assumeCuda() {
        org.junit.jupiter.api.Assumptions.assumeTrue(Engine.onCuda(), "the row-norm variants are chosen on CUDA");
    }
}
