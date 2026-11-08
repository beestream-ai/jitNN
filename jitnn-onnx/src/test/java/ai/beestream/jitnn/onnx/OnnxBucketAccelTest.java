package ai.beestream.jitnn.onnx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Shape buckets ({@link OnnxSession#withBuckets}) on models with symbolic dimensions, exported by
 * scripts/onnx/bucket_models.py into {@code -Djitnn.onnx.bucketModels=<dir>}: requests of many
 * shapes give the same outputs as unpadded runs, and compile one plan per bucket.
 */
class OnnxBucketAccelTest {

    private static Path dir;

    @BeforeAll
    static void models() {
        String d = System.getProperty("jitnn.onnx.bucketModels", "");
        Assumptions.assumeTrue(!d.isEmpty(), "set -Djitnn.onnx.bucketModels to the output of scripts/onnx/bucket_models.py");
        dir = Path.of(d);
    }

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static void close(float[] e, float[] a, double tol, String what) {
        assertEquals(e.length, a.length, what + ": length");
        for (int i = 0; i < e.length; i++) {
            assertTrue(Math.abs(e[i] - a[i]) <= tol * Math.max(1, Math.abs(e[i])), what + ": element " + i + " expected " + e[i] + " got " + a[i]);
        }
    }

    @Test
    void encoderBatchAndSequence() throws Exception {
        OnnxSession exact = OnnxSession.load(dir.resolve("encoder.onnx"));
        OnnxSession bucketed = OnnxSession.load(dir.resolve("encoder.onnx")).withBuckets(4, 8, 16, 32, 64, 128);
        System.out.println("[bucket] encoder pads " + bucketed.paddedDimensions());
        assertTrue(bucketed.paddedDimensions().size() == 2, "batch and sequence are padded: " + bucketed.paddedDimensions());
        Random r = new Random(1);
        long[] before = null;
        for (int req = 0; req < 12; req++) {
            int b = 1 + r.nextInt(8);
            int t = 3 + r.nextInt(60);
            int[] ids = new int[b * t];
            int[] mask = new int[b * t];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = 1 + r.nextInt(999);
                mask[i] = 1;
            }
            Map<String, Tensor> in = Map.of("input_ids", Tensor.of(ids, b, t), "attention_mask", Tensor.of(mask, b, t));
            float[] e = exact.run(in).get("hidden").toFloatArray();
            Tensor got = bucketed.run(in).get("hidden");
            assertEquals(b, got.dim(0), "batch cut back");
            assertEquals(t, got.dim(1), "sequence cut back");
            close(e, got.toFloatArray(), 2e-4, "request " + req + " [" + b + ", " + t + "]");
            if (req == 0) {
                before = Engine.planStats();
            }
        }
        long[] after = Engine.planStats();
        System.out.println("[bucket] plans compiled over 12 requests: " + (after[0] - before[0]));
    }

    @Test
    void cnnBatch() throws Exception {
        OnnxSession exact = OnnxSession.load(dir.resolve("cnn.onnx"));
        OnnxSession bucketed = OnnxSession.load(dir.resolve("cnn.onnx")).withBuckets(1, 2, 4, 8, 16, 32);
        Random r = new Random(2);
        for (int req = 0; req < 10; req++) {
            int b = 1 + r.nextInt(20);
            float[] x = new float[b * 3 * 32 * 32];
            for (int i = 0; i < x.length; i++) {
                x[i] = (float) r.nextGaussian();
            }
            Map<String, Tensor> in = Map.of("images", Tensor.of(x, b, 3, 32, 32));
            float[] e = exact.run(in).get("logits").toFloatArray();
            Tensor got = bucketed.run(in).get("logits");
            assertEquals(b, got.dim(0));
            close(e, got.toFloatArray(), 1e-4, "batch " + b);
        }
    }

    /** A stream of requests of random shapes: plans compiled and time, unpadded against bucketed (after a warm-up of every bucket). */
    @Test
    void mixedShapeStream() throws Exception {
        int[] buckets = { 8, 16, 32, 64, 128 };
        OnnxSession exact = OnnxSession.load(dir.resolve("encoder.onnx"));
        OnnxSession bucketed = OnnxSession.load(dir.resolve("encoder.onnx")).withBuckets(1, 2, 4, 8, 16, 32, 64, 128);
        for (int b : new int[] { 1, 2, 4, 8 }) {
            for (int t : buckets) {
                run(bucketed, b, t, 0);
            }
        }
        Random r = new Random(3);
        int[][] reqs = new int[40][];
        for (int i = 0; i < reqs.length; i++) {
            reqs[i] = new int[] { 1 + r.nextInt(8), 4 + r.nextInt(120) };
        }
        for (OnnxSession s : new OnnxSession[] { exact, bucketed }) {
            long[] p0 = Engine.planStats();
            long t0 = System.nanoTime();
            for (int[] q : reqs) {
                run(s, q[0], q[1], q[0] * 131 + q[1]);
            }
            double sec = (System.nanoTime() - t0) / 1e9;
            long[] p1 = Engine.planStats();
            System.out.printf("[bucket] %s: 40 requests of random shapes in %.2f s, %d plans compiled%n", s == exact ? "unpadded" : "bucketed", sec, p1[0] - p0[0]);
        }
    }

    private static void run(OnnxSession s, int b, int t, int seed) {
        int[] ids = new int[b * t];
        int[] mask = new int[b * t];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = 1 + (i * 31 + seed) % 999;
            mask[i] = 1;
        }
        s.run(Map.of("input_ids", Tensor.of(ids, b, t), "attention_mask", Tensor.of(mask, b, t))).get("hidden").toFloatArray();
    }
}
