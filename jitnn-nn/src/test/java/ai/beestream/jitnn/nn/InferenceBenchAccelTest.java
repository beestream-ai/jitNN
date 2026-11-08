package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Supplier;

import ai.beestream.jitnn.Autocast;
import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.models.Bert;
import ai.beestream.jitnn.nn.models.ResNet;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Inference latency of ResNet-50 and BERT-base, measured the way jitNN-benchmarking's scripts/inference.py measures
 * ONNX Runtime, PyTorch and DJL on the same GPU: host input in, host output out, a warm-up, then
 * the median of timed runs. The models are built from the same seeds as the Python side, so every
 * framework runs the same weights on the same inputs; with {@code -Djitnn.bench.out=<dir>} the
 * outputs are written there for inference.py to compare.
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

    /** The deterministic inputs inference.py builds: pixels in [-1, 1) and token ids spread over the vocabulary. */
    static uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray half(float[] x) {
        var h = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(x.length);
        for (int i = 0; i < x.length; i++) {
            h.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat(x[i]));
        }
        return h;
    }

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

    /**
     * Median milliseconds of {@code RUNS} calls after {@code WARMUP}. A call runs the forward pass and
     * brings its output to the host in its own type (FP16 stays FP16), as PyTorch's
     * {@code model(x).cpu().numpy()} and ONNX Runtime's {@code run} do; the first call's output is
     * converted to floats and saved.
     */
    private static void time(String name, int samples, Supplier<Tensor> forward) throws Exception {
        if (!name.contains(System.getProperty("jitnn.bench.only", ""))) {
            return;
        }
        // as a serving loop does: the output is fetched to the host, read, and released
        Supplier<Object> call = () -> {
            Tensor y = forward.get();
            Engine.sync(y);
            y.release();
            return y;
        };
        long c0 = System.nanoTime();
        float[] first = forward.get().toFloatArray();
        double compile = (System.nanoTime() - c0) / 1e6;
        save(name, first);
        for (int i = 1; i < WARMUP; i++) {
            call.get();
        }
        boolean profile = Boolean.getBoolean("jitnn.profile");
        if (profile) {
            Engine.kernelBreakdown();
        }
        boolean cpuStats = Boolean.getBoolean("jitnn.cpu.stats");
        if (cpuStats) {
            Engine.cpuReport();
        }
        double[] ms = new double[RUNS];
        long[] before = Engine.trafficStats();
        long[] plans = Engine.planStats();
        for (int i = 0; i < RUNS; i++) {
            long t0 = System.nanoTime();
            call.get();
            ms[i] = (System.nanoTime() - t0) / 1e6;
        }
        long[] after = Engine.trafficStats();
        Arrays.sort(ms);
        double median = ms[RUNS / 2];
        System.out.printf("[inference] jitnn %-28s median %8.3f ms  p10 %8.3f  p90 %8.3f  %10.1f samples/s  (first call %.0f ms)%n", name, median, ms[RUNS / 10], ms[RUNS * 9 / 10],
                samples * 1000 / median, compile);
        if (cpuStats) {
            System.out.print(Engine.cpuReport());
        }
        long[] plansAfter = Engine.planStats();
        System.out.printf("[inference] plans %-28s %d compiled, %d replayed (%d remembered) in %d calls%n", name, plansAfter[0] - plans[0], plansAfter[1] - plans[1], plansAfter[3] - plans[3],
                RUNS);
        // per call: kernel launches, library calls, and the bytes the operators read and write at least
        System.out.printf("[traffic] %-28s %6.0f kernels %4.0f library calls %10.1f MB%n", name, (after[1] - before[1]) / (double) RUNS, (after[2] - before[2]) / (double) RUNS,
                (after[3] - before[3]) / 1e6 / RUNS);
        if (profile) {
            printProfile(name, Engine.kernelBreakdown());
        }
        int threads = Integer.getInteger("jitnn.bench.threads", 1);
        if (threads > 1) {
            throughput(name, samples, threads, call);
        }
    }

    /**
     * {@code threads} threads calling at once ({@code -Djitnn.bench.threads}, with
     * {@code -Djitnn.concurrency} slots): samples per second over all of them, and the median latency
     * of a call.
     */
    private static void throughput(String name, int samples, int threads, Supplier<Object> call) throws Exception {
        java.util.concurrent.ExecutorService pool = java.util.concurrent.Executors.newFixedThreadPool(threads);
        try {
            // warm every slot: each compiles its own plans
            java.util.List<java.util.concurrent.Future<?>> warm = new java.util.ArrayList<>();
            for (int t = 0; t < threads; t++) {
                warm.add(pool.submit(() -> {
                    for (int i = 0; i < WARMUP; i++) {
                        call.get();
                    }
                }));
            }
            for (var f : warm) {
                f.get();
            }
            double[] ms = new double[threads * RUNS];
            java.util.concurrent.atomic.AtomicInteger at = new java.util.concurrent.atomic.AtomicInteger();
            long t0 = System.nanoTime();
            java.util.List<java.util.concurrent.Future<?>> runs = new java.util.ArrayList<>();
            for (int t = 0; t < threads; t++) {
                runs.add(pool.submit(() -> {
                    for (int i = 0; i < RUNS; i++) {
                        long c0 = System.nanoTime();
                        call.get();
                        ms[at.getAndIncrement()] = (System.nanoTime() - c0) / 1e6;
                    }
                }));
            }
            for (var f : runs) {
                f.get();
            }
            double wall = (System.nanoTime() - t0) / 1e9;
            Arrays.sort(ms);
            System.out.printf("[inference] jitnn %-28s %d threads: %10.1f samples/s, median latency %8.3f ms%n", name, threads, samples * (double) ms.length / wall, ms[ms.length / 2]);
        } finally {
            pool.shutdown();
        }
    }

    /**
     * Where the wall time of one call goes ({@code -Djitnn.bench.split=true}): recording the forward
     * pass, executing it (the plan, with the output left on the device), fetching the output, and
     * converting it to a float[]; medians of {@code RUNS} calls.
     */
    private static void split(String name, Supplier<Tensor> forward) {
        if (!name.contains(System.getProperty("jitnn.bench.only", ""))) {
            return;
        }
        int n = RUNS;
        double[][] t = new double[6][n];
        for (int i = 0; i < WARMUP + n; i++) {
            long t0 = System.nanoTime();
            Tensor y = forward.get();
            long t1 = System.nanoTime();
            long[] before = Engine.timeSplit();
            Engine.sync();
            long[] after = Engine.timeSplit();
            long t2 = System.nanoTime();
            if (i >= WARMUP) {
                t[4][i - WARMUP] = (after[0] - before[0]) / 1e6;
                t[5][i - WARMUP] = (after[1] - before[1]) / 1e6;
            }
            Engine.sync(y);
            long t3 = System.nanoTime();
            y.toFloatArray();
            long t4 = System.nanoTime();
            // released, as in time(): a dropped output that is still held keeps a copy of its own when
            // the plan runs again
            y.release();
            if (i >= WARMUP) {
                t[0][i - WARMUP] = (t1 - t0) / 1e6;
                t[1][i - WARMUP] = (t2 - t1) / 1e6;
                t[2][i - WARMUP] = (t3 - t2) / 1e6;
                t[3][i - WARMUP] = (t4 - t3) / 1e6;
            }
        }
        for (double[] a : t) {
            Arrays.sort(a);
        }
        System.out.printf("[inference] split %-28s record %.3f  execute %.3f (jitNN host %.3f, TornadoVM execute() %.3f)  fetch %.3f  convert %.3f ms (median)%n", name, t[0][n / 2], t[1][n / 2],
                t[4][n / 2], t[5][n / 2], t[2][n / 2], t[3][n / 2]);
    }

    /** Device time and launches per call, by operator (with {@code -Djitnn.profile=true}; on CUDA also {@code -Djitnn.cudaGraph=false}). */
    private static void printProfile(String name, java.util.Map<String, long[]> perTask) {
        if (Boolean.getBoolean("jitnn.bench.profileTasks")) {
            // every task of the plan, not grouped by operator
            perTask.entrySet().stream().sorted((x, y) -> Long.compare(y.getValue()[1], x.getValue()[1])).limit(20)
                    .forEach(en -> System.out.printf("[inference]   task %-44s %8.3f ms  %4d launches%n", en.getKey(), en.getValue()[1] / 1e6 / RUNS, en.getValue()[0] / RUNS));
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
        System.out.printf("[inference] profile %s: %d launches, %.3f ms of kernels per call%n", name, launches / RUNS, total / 1e6 / RUNS);
        long all = total;
        perOp.entrySet().stream().sorted((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1])).limit(12)
                .forEach(en -> System.out.printf("[inference]   %-40s %8.3f ms %5.1f%%  %4d launches%n", en.getKey(), en.getValue()[1] / 1e6 / RUNS,
                        100.0 * en.getValue()[1] / all, en.getValue()[0] / RUNS));
    }

    @Test
    void resnet50() throws Exception {
        if (!enabled("resnet50")) {
            return;
        }
        long m0 = System.nanoTime();
        Module model = ResNet.resnet50(3, 1000, false, 1);
        System.out.printf("[inference] build resnet50: %.0f ms%n", (System.nanoTime() - m0) / 1e6);
        model.train(false);
        // fp32; fp16h: FP16 activations, convolutions on cuDNN in FP16 (needs -Djitnn.native)
        for (String mode : new String[] { "fp32", "fp16h" }) {
            for (int b : new int[] { 1, 32, 128 }) {
                float[] x = images(b * 3 * 224 * 224);
                // FP16 runs take an FP16 image, as PyTorch and ONNX Runtime do in inference.py
                uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray xh = mode.equals("fp16h") ? half(x) : null;
                String name = "resnet50_b" + b + "_" + mode;
                Supplier<Tensor> forward = () -> Grad.inference(() -> {
                    try (Grad.Scope ignored = mode.equals("fp16h") ? Autocast.enableHalf() : null) {
                        return model.forward(xh != null ? Tensor.of(xh, b, 3, 224, 224) : Tensor.wrap(x, b, 3, 224, 224));
                    }
                });
                if (Boolean.getBoolean("jitnn.bench.split")) {
                    split(name, forward);
                }
                time(name, b, forward);
            }
        }
        Engine.reset();
    }

    @Test
    void bertBase() throws Exception {
        if (!enabled("bert")) {
            return;
        }
        long m0 = System.nanoTime();
        Bert model = new Bert(BERT_VOCAB, 512, 768, 12, 12, 1);
        System.out.printf("[inference] build bert: %.0f ms%n", (System.nanoTime() - m0) / 1e6);
        model.train(false);
        int[][] shapes = { { 1, 128 }, { 32, 128 }, { 8, 512 } };
        // fp32; fp16: autocast (FP16 weights and products, FP32 activations); fp16h: FP16 activations too
        for (String mode : new String[] { "fp32", "fp16", "fp16h" }) {
            for (int[] s : shapes) {
                int b = s[0];
                int t = s[1];
                int[] ids = tokens(b * t);
                String name = "bert_b" + b + "_s" + t + "_" + mode;
                if (Boolean.getBoolean("jitnn.bench.split")) {
                    split(name, () -> Grad.inference(() -> {
                        try (Grad.Scope ignored = mode.equals("fp16") ? Autocast.enable(DType.F16) : mode.equals("fp16h") ? Autocast.enableHalf() : null) {
                            return model.forward(Tensor.wrap(ids, b, t));
                        }
                    }));
                }
                time(name, b, () -> Grad.inference(() -> {
                    try (Grad.Scope ignored = mode.equals("fp16") ? Autocast.enable(DType.F16) : mode.equals("fp16h") ? Autocast.enableHalf() : null) {
                        return model.forward(Tensor.wrap(ids, b, t));
                    }
                }));
            }
        }
        Engine.reset();
    }
}
