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
}
