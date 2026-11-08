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
package ai.beestream.jitnn.nn;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ai.beestream.jitnn.Autocast;
import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.models.GPT2;
import ai.beestream.jitnn.nn.models.Llama;
import ai.beestream.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Where a transformer training step spends its time: wall time per step and, with
 * {@code -Djitnn.test.profile=true}, device time per operator. Random tokens; sizes from
 * {@code -Djitnn.bench.dim}, {@code .layers}, {@code .heads} (dim / 32 by default), {@code .seq}, {@code .batch}.
 */
class TransformerBenchAccelTest {

    static final int DIM = Integer.getInteger("jitnn.bench.dim", 128);
    static final int LAYERS = Integer.getInteger("jitnn.bench.layers", 4);
    static final int SEQ = Integer.getInteger("jitnn.bench.seq", 128);
    static final int BATCH = Integer.getInteger("jitnn.bench.batch", 16);
    static final int HEADS = Integer.getInteger("jitnn.bench.heads", Math.max(1, DIM / 32));
    static final int WARM = 5;
    static final int STEPS = 20;

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    @Test
    void gpt2() {
        bench("GPT-2", new GPT2(256, SEQ, DIM, HEADS, LAYERS, 1), null);
        bench("GPT-2 BF16", new GPT2(256, SEQ, DIM, HEADS, LAYERS, 1), DType.BF16);
        bench("GPT-2 FP16 autocast", new GPT2(256, SEQ, DIM, HEADS, LAYERS, 1), DType.F16);
        bench("GPT-2 FP16 activations", new GPT2(256, SEQ, DIM, HEADS, LAYERS, 1), DType.F16, true);
    }

    @Test
    void llama() {
        bench("Llama", new Llama(256, DIM, HEADS, LAYERS, DIM * 8 / 3 / 32 * 32, 10000.0f, 1), null);
    }

    private static void bench(String name, Module model, DType autocast) {
        bench(name, model, autocast, false);
    }

    /** {@code half}: FP16 activations end to end ({@link Autocast#enableHalf}); FP16 runs use a loss scaler. */
    private static void bench(String name, Module model, DType autocast, boolean half) {
        ai.beestream.jitnn.nn.optim.GradScaler scaler = autocast == DType.F16 ? new ai.beestream.jitnn.nn.optim.GradScaler() : null;
        List<Tensor> params = model.parameters();
        Adam opt = Adam.adamW(params, 1e-3f, 0.01f);
        int[] x = new int[BATCH * SEQ];
        int[] y = new int[BATCH * SEQ];
        java.util.Random r = new java.util.Random(7);
        long t0 = 0;
        for (int s = 0; s < WARM + STEPS; s++) {
            if (s == WARM) {
                JitnnRuntime.kernelBreakdown();
                t0 = System.nanoTime();
            }
            for (int i = 0; i < x.length; i++) {
                x[i] = r.nextInt(256);
                y[i] = r.nextInt(256);
            }
            Tensor loss;
            try (Grad.Scope ignored = half ? Autocast.enableHalf() : autocast != null ? Autocast.enable(autocast) : null) {
                loss = F.crossEntropy(model.forward(Tensor.of(x, BATCH, SEQ)).reshape(BATCH * SEQ, -1), Tensor.of(y, BATCH * SEQ));
            }
            if (scaler != null) {
                scaler.scale(loss).backward();
                scaler.unscale(opt);
                opt.clipGradNorm(1.0f);
                scaler.step(opt);
                loss.item();
                scaler.update();
            } else {
                loss.backward();
                opt.clipGradNorm(1.0f);
                opt.step();
                loss.item();
            }
            opt.zeroGrad();
        }
        double wall = (System.nanoTime() - t0) / 1e6 / STEPS;
        Map<String, long[]> perTask = JitnnRuntime.kernelBreakdown();
        Map<String, long[]> perOp = new TreeMap<>();
        long total = 0;
        long launches = 0;
        for (var en : perTask.entrySet()) {
            String op = en.getKey().replaceAll("[\\[#].*", "");
            long[] a = perOp.computeIfAbsent(op, k -> new long[2]);
            a[0] += en.getValue()[0];
            a[1] += en.getValue()[1];
            total += en.getValue()[1];
            launches += en.getValue()[0];
        }
        System.out.printf("[bench] %s (dim %d, %d layers, %d x %d tokens): %.1f ms/step wall, %.0f MB of intermediates", name, DIM, LAYERS, BATCH, SEQ, wall,
                JitnnRuntime.lastIntermediateBytes() / 1e6);
        if (total > 0) {
            System.out.printf(", device kernels %.1f ms in %d launches (%.0f%%)%n", total / 1e6 / STEPS, launches / STEPS, 100.0 * total / 1e6 / STEPS / wall);
            long all = total;
            String detail = System.getProperty("jitnn.bench.tasks", "");
            if (!detail.isEmpty()) {
                perTask.entrySet().stream().filter(en -> en.getKey().contains(detail)).forEach(en -> System.out.printf("[bench]   task %-36s %6.2f ms%n", en.getKey(), en.getValue()[1] / 1e6 / STEPS));
            }
            boolean shapes = Boolean.getBoolean("jitnn.bench.shapes");
            Map<String, long[]> rows = shapes ? perShape(perTask) : perOp;
            rows.entrySet().stream().sorted((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1])).limit(shapes ? 24 : 14)
                    .forEach(en -> System.out.printf("[bench]   %-44s %6.2f ms %5.1f%%  %4d launches%n", en.getKey(), en.getValue()[1] / 1e6 / STEPS,
                            100.0 * en.getValue()[1] / all, en.getValue()[0] / STEPS));
        } else {
            System.out.println();
        }
    }

    /** Totals per operator and shape (the task index dropped). */
    private static Map<String, long[]> perShape(Map<String, long[]> perTask) {
        Map<String, long[]> out = new TreeMap<>();
        for (var en : perTask.entrySet()) {
            long[] a = out.computeIfAbsent(en.getKey().substring(0, en.getKey().indexOf('#')), k -> new long[2]);
            a[0] += en.getValue()[0];
            a[1] += en.getValue()[1];
        }
        return out;
    }
}
