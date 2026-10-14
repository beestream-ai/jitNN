package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Supplier;

import org.beehive.jitnn.Autocast;
import org.beehive.jitnn.DType;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.models.Bert;
import org.beehive.jitnn.nn.models.ResNet;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Inference latency of ResNet-50 and BERT-base, measured the way scripts/bench/t1.py measures
 * ONNX Runtime, PyTorch and DJL on the same GPU: host input in, host output out, a warm-up, then
 * the median of timed runs. The models are built from the same seeds as the Python side, so every
 * framework runs the same weights on the same inputs; with {@code -Djitnn.bench.out=<dir>} the
 * outputs are written there for t1.py to compare.
 *
 * <p>Runs only with {@code -Djitnn.bench.inference=true}; {@code -Djitnn.bench.models=resnet50,bert}
 * picks the models.
 */
class InferenceBenchAccelTest {

    static final int WARMUP = Integer.getInteger("jitnn.bench.warmup", 10);
    static final int RUNS = Integer.getInteger("jitnn.bench.runs", 50);
    static final int BERT_VOCAB = 30522;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static boolean enabled(String model) {
        assumeTrue(Boolean.getBoolean("jitnn.bench.inference"), "set -Djitnn.bench.inference=true");
        return Arrays.asList(System.getProperty("jitnn.bench.models", "resnet50,bert").split(",")).contains(model);
    }

    /** The deterministic inputs t1.py builds: pixels in [-1, 1) and token ids spread over the vocabulary. */
    static float[] images(int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) ((i * 2654435761L) % 1000) / 500.0f - 1.0f;
        }
        return x;
    }

    static int[] tokens(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = (int) ((i * 7919L + 13) % BERT_VOCAB);
        }
        return t;
    }

    private static void save(String name, float[] out) throws Exception {
        String dir = System.getProperty("jitnn.bench.out", "");
        if (dir.isEmpty()) {
            return;
        }
        ByteBuffer b = ByteBuffer.allocate(out.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(out);
        Files.write(Path.of(dir, name + ".jitnn.f32"), b.array());
    }

    /** Median milliseconds of {@code RUNS} calls after {@code WARMUP}; the first call's output is saved. */
    private static void time(String name, int samples, Supplier<float[]> call) throws Exception {
        long c0 = System.nanoTime();
        float[] first = call.get();
        double compile = (System.nanoTime() - c0) / 1e6;
        save(name, first);
        for (int i = 1; i < WARMUP; i++) {
            call.get();
        }
        boolean profile = Boolean.getBoolean("jitnn.profile");
        if (profile) {
            Engine.kernelBreakdown();
        }
        double[] ms = new double[RUNS];
        for (int i = 0; i < RUNS; i++) {
            long t0 = System.nanoTime();
            call.get();
            ms[i] = (System.nanoTime() - t0) / 1e6;
        }
        Arrays.sort(ms);
        double median = ms[RUNS / 2];
        System.out.printf("[t1] jitnn %-28s median %8.3f ms  p10 %8.3f  p90 %8.3f  %10.1f samples/s  (first call %.0f ms)%n", name, median, ms[RUNS / 10], ms[RUNS * 9 / 10],
                samples * 1000 / median, compile);
        if (profile) {
            printProfile(name, Engine.kernelBreakdown());
        }
    }

    /** Device time and launches per call, by operator (with {@code -Djitnn.profile=true}; on CUDA also {@code -Djitnn.cudaGraph=false}). */
    private static void printProfile(String name, java.util.Map<String, long[]> perTask) {
        if (Boolean.getBoolean("jitnn.bench.profileTasks")) {
            // every task of the plan, not grouped by operator
            perTask.entrySet().stream().sorted((x, y) -> Long.compare(y.getValue()[1], x.getValue()[1])).limit(20)
                    .forEach(en -> System.out.printf("[t1]   task %-44s %8.3f ms  %4d launches%n", en.getKey(), en.getValue()[1] / 1e6 / RUNS, en.getValue()[0] / RUNS));
        }
        java.util.Map<String, long[]> perOp = new java.util.TreeMap<>();
        long total = 0;
        long launches = 0;
        for (var en : perTask.entrySet()) {
            long[] a = perOp.computeIfAbsent(en.getKey().replaceAll("[\\[#].*", ""), k -> new long[2]);
            a[0] += en.getValue()[0];
            a[1] += en.getValue()[1];
            total += en.getValue()[1];
            launches += en.getValue()[0];
        }
        System.out.printf("[t1] profile %s: %d launches, %.3f ms of kernels per call%n", name, launches / RUNS, total / 1e6 / RUNS);
        long all = total;
        perOp.entrySet().stream().sorted((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1])).limit(12)
                .forEach(en -> System.out.printf("[t1]   %-40s %8.3f ms %5.1f%%  %4d launches%n", en.getKey(), en.getValue()[1] / 1e6 / RUNS,
                        100.0 * en.getValue()[1] / all, en.getValue()[0] / RUNS));
    }

    @Test
    void resnet50() throws Exception {
        if (!enabled("resnet50")) {
            return;
        }
        Module model = ResNet.resnet50(3, 1000, false, 1);
        model.train(false);
        for (int b : new int[] { 1, 32, 128 }) {
            float[] x = images(b * 3 * 224 * 224);
            time("resnet50_b" + b + "_fp32", b, () -> Grad.inference(() -> model.forward(Tensor.of(x, b, 3, 224, 224))).toFloatArray());
        }
        Engine.reset();
    }

    @Test
    void bertBase() throws Exception {
        if (!enabled("bert")) {
            return;
        }
        Bert model = new Bert(BERT_VOCAB, 512, 768, 12, 12, 1);
        model.train(false);
        int[][] shapes = { { 1, 128 }, { 32, 128 }, { 8, 512 } };
        // fp32; fp16: autocast (FP16 weights and products, FP32 activations); fp16h: FP16 activations too
        for (String mode : new String[] { "fp32", "fp16", "fp16h" }) {
            for (int[] s : shapes) {
                int b = s[0];
                int t = s[1];
                int[] ids = tokens(b * t);
                String name = "bert_b" + b + "_s" + t + "_" + mode;
                time(name, b, () -> Grad.inference(() -> {
                    try (Grad.Scope ignored = mode.equals("fp16") ? Autocast.enable(DType.F16) : mode.equals("fp16h") ? Autocast.enableHalf() : null) {
                        return model.forward(Tensor.of(ids, b, t));
                    }
                }).toFloatArray());
            }
        }
        Engine.reset();
    }
}
