package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.beehive.jitnn.Buckets;
import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.models.Bert;
import org.beehive.jitnn.nn.models.GPT2;
import org.beehive.jitnn.nn.models.ResNet;
import org.beehive.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Requests whose shapes change from one call to the next, as a server or a ragged data set sends
 * them: GPT-2 and BERT inference over random batch x sequence, ResNet-18 inference over random
 * batches, and MLP training over random batch sizes. For each: plans compiled, the time spent in
 * the calls that compiled, the median of the calls that did not, the total, and the largest
 * intermediate memory. The shapes come from a fixed seed, so every run sees the same sequence.
 *
 * <p>Each scenario runs twice: on the exact shapes, then padded to {@link Buckets} through
 * {@link Bucketing} (batch in powers of two, sequence length in multiples of 64), with the same
 * shapes.
 *
 * <p>Runs only with {@code -Djitnn.bench.dynamic=true}; {@code -Djitnn.bench.dynamic.requests}
 * (default 40) sets the number of requests per scenario, and {@code -Djitnn.bench.dynamic.only}
 * runs only the scenarios whose name contains it.
 */
class DynamicShapeBenchAccelTest {

    static final int REQUESTS = Integer.getInteger("jitnn.bench.dynamic.requests", 40);

    @AfterAll
    static void release() {
        Engine.reset();
    }

    /** One request: runs it and returns a result, read back so the call completes. */
    interface Request {
        float[] run(int i);
    }

    private static void measure(String name, Request request) {
        if (!name.contains(System.getProperty("jitnn.bench.dynamic.only", ""))) {
            return;
        }
        long[] before = Engine.planStats();
        List<Double> compiling = new ArrayList<>();
        List<Double> steady = new ArrayList<>();
        long peak = 0;
        long t0 = System.nanoTime();
        for (int i = 0; i < REQUESTS; i++) {
            long c0 = Engine.planStats()[0];
            long r0 = System.nanoTime();
            try {
                request.run(i);
            } catch (RuntimeException e) {
                throw new IllegalStateException(name + ": request " + i + " failed", e);
            }
            double ms = (System.nanoTime() - r0) / 1e6;
            (Engine.planStats()[0] > c0 ? compiling : steady).add(ms);
            peak = Math.max(peak, Engine.lastIntermediateBytes());
        }
        double total = (System.nanoTime() - t0) / 1e9;
        long[] after = Engine.planStats();
        double[] s = steady.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        System.out.printf("[dynamic] %-47s %3d requests: %3d plans compiled, %3d replayed; %6.1f s total, %6.1f s in compiling calls, steady median %7.2f ms (%d calls), peak intermediates %.0f MB%n",
                name, REQUESTS, after[0] - before[0], after[1] - before[1], total, compiling.stream().mapToDouble(Double::doubleValue).sum() / 1e3,
                s.length == 0 ? Double.NaN : s[s.length / 2], s.length, peak / 1e6);
        Engine.reset();
    }

    private static Tensor tokens(Random r, int b, int t, int vocab) {
        int[] ids = new int[b * t];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = r.nextInt(vocab);
        }
        return Tensor.of(ids, b, t);
    }

    /** Uniform values in [-1, 1) made on the host, so a new request never compiles a generator plan. */
    private static Tensor host(Random r, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = 2 * r.nextFloat() - 1;
        }
        return Tensor.of(v, dims);
    }

    @Test
    void changingShapes() {
        assumeTrue(Boolean.getBoolean("jitnn.bench.dynamic"), "set -Djitnn.bench.dynamic=true");
        // room for every bucket of a scenario (GPT-2 and BERT: 5 batch x 4 sequence sizes)
        Engine.reservePlans(32);
        int vocab = 1000, dim = 256, heads = 4, layers = 4, context = 256;
        Buckets batch16 = Buckets.powersOfTwo(1, 16), seq = Buckets.multiplesOf(64, context);

        GPT2 gpt2 = new GPT2(vocab, context, dim, heads, layers, 1);
        for (boolean bucketed : new boolean[] { false, true }) {
            Random rg = new Random(11);
            measure("GPT-2 inference, batch 1-16 x seq 8-256" + (bucketed ? ", buckets" : ""), i -> {
                Tensor idx = tokens(rg, 1 + rg.nextInt(16), 8 + rg.nextInt(249), vocab);
                return Grad.inference(() -> bucketed ? Bucketing.causal(batch16, seq, idx, 0, gpt2::forward) : gpt2.forward(idx)).toFloatArray();
            });
        }

        Bert bert = new Bert(vocab, context, dim, heads, layers, 2);
        for (boolean bucketed : new boolean[] { false, true }) {
            Random rb = new Random(12);
            measure("BERT inference, batch 1-16 x seq 8-256" + (bucketed ? ", buckets" : ""), i -> {
                int b = 1 + rb.nextInt(16), t = 8 + rb.nextInt(249);
                // a ragged batch: each row's length up to t, the rest masked out
                float[] mask = new float[b * t];
                for (int row = 0; row < b; row++) {
                    int len = 1 + rb.nextInt(t);
                    Arrays.fill(mask, row * t + len, (row + 1) * t, -1e9f);
                }
                Tensor idx = tokens(rb, b, t, vocab);
                Tensor m = Tensor.of(mask, b, 1, 1, t);
                return Grad.inference(() -> bucketed ? Bucketing.masked(batch16, seq, idx, m, 0, bert::encode) : bert.encode(idx, m)).toFloatArray();
            });
        }

        Module resnet = ResNet.resnet18(3, 10, true, 3).train(false);
        Buckets batch64 = Buckets.powersOfTwo(1, 64);
        for (boolean bucketed : new boolean[] { false, true }) {
            Random rr = new Random(13);
            measure("ResNet-18 inference, batch 1-64, 32x32" + (bucketed ? ", buckets" : ""), i -> {
                int b = 1 + rr.nextInt(64);
                Tensor x = host(rr, b, 3, 32, 32);
                return Grad.inference(() -> bucketed ? Bucketing.rows(batch64, x, resnet::forward) : resnet.forward(x)).toFloatArray();
            });
        }

        Buckets batch128 = Buckets.powersOfTwo(1, 128);
        for (boolean bucketed : new boolean[] { false, true }) {
            Sequential mlp = new Sequential(new Linear(64, 256, 4), new ReLU(), new Linear(256, 256, 5), new ReLU(), new Linear(256, 10, 6));
            Adam opt = new Adam(mlp.parameters(), 1e-3f);
            Random rm = new Random(14);
            measure("MLP training step, batch 1-128" + (bucketed ? ", buckets" : ""), i -> {
                int b = 1 + rm.nextInt(128);
                Tensor x = host(rm, b, 64);
                int[] y = new int[b];
                for (int k = 0; k < b; k++) {
                    y[k] = rm.nextInt(10);
                }
                Tensor labels = Tensor.of(y, b);
                Tensor loss = bucketed ? Bucketing.crossEntropy(mlp.forward(Buckets.pad(x, 0, batch128.size(b), 0)), labels)
                        : F.crossEntropy(mlp.forward(x), labels);
                loss.backward();
                opt.step();
                opt.zeroGrad();
                return new float[] { loss.item() };
            });
        }
    }
    /** The median time of {@code reps} runs after two warm-up runs. */
    private static double median(int reps, Request request) {
        request.run(0);
        request.run(0);
        double[] t = new double[reps];
        for (int i = 0; i < reps; i++) {
            long t0 = System.nanoTime();
            request.run(0);
            t[i] = (System.nanoTime() - t0) / 1e6;
        }
        Arrays.sort(t);
        return t[reps / 2];
    }

    private static void cost(String name, double[] exact, double[] padded) {
        double e = 0, p = 0;
        for (int i = 0; i < exact.length; i++) {
            e += exact[i];
            p += padded[i];
        }
        System.out.printf("[padding] %-40s exact %s ms, padded %s ms: %.2fx%n", name, Arrays.toString(round(exact)), Arrays.toString(round(padded)), p / e);
    }

    private static double[] round(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = Math.round(v[i] * 100) / 100.0;
        }
        return r;
    }

    /**
     * What padding costs once every plan is compiled: per shape, the median time on the exact shape
     * against the same request padded to its bucket. Runs with {@code -Djitnn.bench.padding=true}.
     */
    @Test
    void paddingCost() {
        assumeTrue(Boolean.getBoolean("jitnn.bench.padding"), "set -Djitnn.bench.padding=true");
        Engine.reservePlans(32);
        int vocab = 1000, dim = 256, heads = 4, layers = 4, context = 256, reps = 15;
        Buckets batch16 = Buckets.powersOfTwo(1, 16), seq = Buckets.multiplesOf(64, context);
        int[][] tokenShapes = { { 1, 40 }, { 3, 100 }, { 5, 130 }, { 9, 70 }, { 12, 200 }, { 16, 250 } };
        Random r = new Random(21);

        GPT2 gpt2 = new GPT2(vocab, context, dim, heads, layers, 1);
        double[] e = new double[tokenShapes.length], p = new double[tokenShapes.length];
        for (int k = 0; k < tokenShapes.length; k++) {
            Tensor idx = tokens(r, tokenShapes[k][0], tokenShapes[k][1], vocab);
            e[k] = median(reps, i -> Grad.inference(() -> gpt2.forward(idx)).toFloatArray());
            p[k] = median(reps, i -> Grad.inference(() -> Bucketing.causal(batch16, seq, idx, 0, gpt2::forward)).toFloatArray());
        }
        cost("GPT-2 " + Arrays.deepToString(tokenShapes), e, p);
        Engine.reset();

        Bert bert = new Bert(vocab, context, dim, heads, layers, 2);
        for (int k = 0; k < tokenShapes.length; k++) {
            Tensor idx = tokens(r, tokenShapes[k][0], tokenShapes[k][1], vocab);
            Tensor m = Tensor.of(new float[idx.numel()], tokenShapes[k][0], 1, 1, tokenShapes[k][1]);
            e[k] = median(reps, i -> Grad.inference(() -> bert.encode(idx, m)).toFloatArray());
            p[k] = median(reps, i -> Grad.inference(() -> Bucketing.masked(batch16, seq, idx, m, 0, bert::encode)).toFloatArray());
        }
        cost("BERT", e, p);
        Engine.reset();

        Module resnet = ResNet.resnet18(3, 10, true, 3).train(false);
        Buckets batch64 = Buckets.powersOfTwo(1, 64);
        int[] batches = { 1, 3, 5, 9, 17, 33, 48 };
        double[] er = new double[batches.length], pr = new double[batches.length];
        for (int k = 0; k < batches.length; k++) {
            Tensor x = host(r, batches[k], 3, 32, 32);
            er[k] = median(reps, i -> Grad.inference(() -> resnet.forward(x)).toFloatArray());
            pr[k] = median(reps, i -> Grad.inference(() -> Bucketing.rows(batch64, x, resnet::forward)).toFloatArray());
        }
        cost("ResNet-18 " + Arrays.toString(batches), er, pr);
        Engine.reset();
    }
    /**
     * Device time per operator task of ResNet-18 at batch 17 over 10 runs, largest first. Runs with
     * {@code -Djitnn.bench.profile=true -Djitnn.profile=true -Djitnn.cudaGraph=false}.
     */
    @Test
    void profileResNet() {
        assumeTrue(Boolean.getBoolean("jitnn.bench.profile"), "set -Djitnn.bench.profile=true");
        Module resnet = ResNet.resnet18(3, 10, true, 3).train(false);
        // other batch sizes first, as a server sees them
        for (int b : new int[] { 1, 48 }) {
            Tensor other = host(new Random(b), b, 3, 32, 32);
            Grad.inference(() -> resnet.forward(other)).toFloatArray();
        }
        Tensor x = host(new Random(5), 17, 3, 32, 32);
        for (int i = 0; i < 3; i++) {
            Grad.inference(() -> resnet.forward(x)).toFloatArray();
        }
        Engine.kernelBreakdown();
        for (int i = 0; i < 10; i++) {
            Grad.inference(() -> resnet.forward(x)).toFloatArray();
        }
        var rows = new ArrayList<>(Engine.kernelBreakdown().entrySet());
        rows.sort((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1]));
        long total = rows.stream().mapToLong(r -> r.getValue()[1]).sum();
        System.out.printf("[profile] total %.3f ms per run%n", total / 1e7);
        for (var r : rows.subList(0, Math.min(25, rows.size()))) {
            System.out.printf("[profile] %-24s %6d launches %8.3f ms per run%n", r.getKey(), r.getValue()[0], r.getValue()[1] / 1e7);
        }
        Engine.reset();
    }
    /** Steady ResNet-18 wall time per batch: 20 warm-up runs, then the median of 50. Runs with {@code -Djitnn.bench.steady=true}. */
    @Test
    void steadyResNet() {
        assumeTrue(Boolean.getBoolean("jitnn.bench.steady"), "set -Djitnn.bench.steady=true");
        Module resnet = ResNet.resnet18(3, 10, true, 3).train(false);
        for (int b : new int[] { 1, 17, 48 }) {
            Tensor x = host(new Random(b), b, 3, 32, 32);
            for (int i = 0; i < 20; i++) {
                Grad.inference(() -> resnet.forward(x)).toFloatArray();
            }
            double[] t = new double[50];
            for (int i = 0; i < t.length; i++) {
                long t0 = System.nanoTime();
                Grad.inference(() -> resnet.forward(x)).toFloatArray();
                t[i] = (System.nanoTime() - t0) / 1e6;
            }
            Arrays.sort(t);
            System.out.printf("[steady] ResNet-18 batch %2d: median %.3f ms, min %.3f ms%n", b, t[25], t[0]);
        }
        Engine.reset();
    }
}
