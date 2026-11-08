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
package ai.beestream.jitnn.cpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Locale;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.Test;

/**
 * GFLOP/s of the CPU backend's GEMM: square products, the GEMMs of BERT-base and the convolutions
 * of ResNet-50, at {@code jitnn.cpu.threads}. Not part of the default suite; run it with
 * {@code -Dtest=CpuGemmBench}.
 */
class CpuGemmBench {

    private static final int RUNS = Integer.getInteger("jitnn.bench.runs", 10);

    @Test
    void gemmThroughput() {
        System.out.printf(Locale.ROOT, "[gemm] %d threads, %d FP32 lanes, micro-tile %dx%d%n", Cpu.threads(), Cpu.lanes(), Gemm.MR, Gemm.NR);
        int[][] shapes = { { 256, 256, 256 }, { 512, 512, 512 }, { 1024, 1024, 1024 }, { 2048, 2048, 2048 },
                // BERT-base, 128 tokens: QKV, attention output, FFN up, FFN down
                { 128, 2304, 768 }, { 128, 768, 768 }, { 128, 3072, 768 }, { 128, 768, 3072 },
                // BERT-base, 32 x 128 tokens
                { 4096, 3072, 768 }, { 4096, 768, 3072 } };
        String only = System.getProperty("jitnn.bench.shapes");
        if (only != null) {
            // e.g. 4096x3072x768,128x768x768
            String[] parts = only.split(",");
            shapes = new int[parts.length][];
            for (int i = 0; i < parts.length; i++) {
                String[] d = parts[i].trim().split("x");
                shapes[i] = new int[] { Integer.parseInt(d[0]), Integer.parseInt(d[1]), Integer.parseInt(d[2]) };
            }
            // warm the JIT on the first shape
            raw(shapes[0][0], shapes[0][1], shapes[0][2]);
        }
        for (int[] s : shapes) {
            raw(s[0], s[1], s[2]);
        }
    }

    private static void raw(int m, int n, int k) {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment a = random(arena, (long) m * k, 1);
            MemorySegment b = random(arena, (long) k * n, 2);
            MemorySegment c = arena.allocate((long) m * n * 4, 64);
            // B is a weight: packed once, as the operators cache it
            float[] bp = Gemm.packB(new Gemm.Matrix(b, 0, 1, n), n, k, new float[Gemm.packedSize(n, k, Gemm.NR)]);
            Runnable call = () -> {
                float[] ap = Gemm.packA(new Gemm.Matrix(a, 0, k, 1), m, k, Gemm.scratch(0, Gemm.packedSize(m, k, Gemm.MR)));
                Gemm.compute(ap, bp, m, n, k, c, 0, n, null, 0, 0);
            };
            report(String.format(Locale.ROOT, "%dx%dx%d", m, n, k), 2.0 * m * n * k, call);
        }
    }

    /**
     * The CPU backend's GEMM next to the native BLAS of {@code jitnn.cpu.blas} on the same products:
     * square, BERT-base (inference and training at 8 x 128 tokens) and the per-image GEMMs of
     * ResNet-50's convolutions (output channels x pixels x input channels times kernel area).
     */
    @Test
    void blasThroughput() {
        org.junit.jupiter.api.Assumptions.assumeTrue(Blas.usable(), "no BLAS library");
        System.out.printf(Locale.ROOT, "[blas] %s, %d threads%n", Blas.library(), Cpu.threads());
        int[][] shapes = { { 512, 512, 512 }, { 1024, 1024, 1024 }, { 2048, 2048, 2048 }, { 128, 768, 768 }, { 128, 3072, 768 }, { 1024, 2304, 768 }, { 1024, 3072, 768 }, { 1024, 768, 3072 },
                { 768, 768, 1024 }, { 4096, 3072, 768 }, { 64, 3136, 576 }, { 256, 3136, 64 }, { 64, 3136, 256 }, { 128, 784, 1152 }, { 256, 196, 2304 }, { 512, 49, 4608 }, { 2048, 49, 512 } };
        for (int[] s : shapes) {
            int m = s[0], n = s[1], k = s[2];
            try (Arena arena = Arena.ofShared()) {
                MemorySegment a = random(arena, (long) m * k, 1);
                MemorySegment b = random(arena, (long) k * n, 2);
                MemorySegment c = arena.allocate((long) m * n * 4, 64);
                MemorySegment d = arena.allocate((long) m * n * 4, 64);
                Gemm.Matrix am = new Gemm.Matrix(a, 0, k, 1);
                Gemm.Matrix bm = new Gemm.Matrix(b, 0, 1, n);
                float[] bp = Gemm.packB(bm, n, k, new float[Gemm.packedSize(n, k, Gemm.NR)]);
                String what = String.format(Locale.ROOT, "%dx%dx%d", m, n, k);
                report(what + " java", 2.0 * m * n * k, () -> {
                    float[] ap = Gemm.packA(am, m, k, Gemm.scratch(0, Gemm.packedSize(m, k, Gemm.MR)));
                    Gemm.compute(ap, bp, m, n, k, c, 0, n, null, 0, 0);
                });
                report(what + " blas", 2.0 * m * n * k, () -> Blas.sgemm(am, bm, m, n, k, d, 0, n, 0f));
                float diff = 0;
                for (long i = 0; i < (long) m * n; i++) {
                    diff = Math.max(diff, Math.abs(c.getAtIndex(ValueLayout.JAVA_FLOAT, i) - d.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
                }
                if (diff > 1e-3f * Math.sqrt(k)) {
                    throw new AssertionError(what + ": BLAS differs by " + diff);
                }
            }
        }
    }

    /** The micro-kernel alone on L1-resident panels: the ceiling the GEMM can reach on one core. */
    @Test
    void kernelOnly() {
        int kc = 256;
        float[] ap = new float[kc * Gemm.MR];
        float[] bp = new float[kc * Gemm.NR];
        java.util.Arrays.fill(ap, 0.5f);
        java.util.Arrays.fill(bp, 0.25f);
        float[] edge = new float[Gemm.MR * Gemm.NR];
        try (Arena arena = Arena.ofShared()) {
            MemorySegment c = arena.allocate(4L * Gemm.MR * Gemm.NR, 64);
            int calls = 200_000;
            for (int rep = 0; rep < 5; rep++) {
                long s = System.nanoTime();
                for (int i = 0; i < calls; i++) {
                    Gemm.kernel(ap, 0, bp, 0, kc, c, 0, Gemm.NR, 0, 0, Gemm.MR, Gemm.NR, (i & 1) == 0, edge);
                }
                double sec = (System.nanoTime() - s) / 1e9;
                System.out.printf(Locale.ROOT, "  kernel %dx%d kc %d: %8.1f GFLOP/s%n", Gemm.MR, Gemm.NR, kc, 2.0 * Gemm.MR * Gemm.NR * kc * calls / sec / 1e9);
            }
        }
    }

    /** ResNet-50's global average pool at batch 32 ([32, 2048, 7, 7] to [32, 2048]). */
    @Test
    void globalAveragePool() {
        Engine.with(Backend.CPU, () -> {
            Tensor x = Tensor.uniform(1, -1, 1, 32, 2048, 7, 7);
            x.toFloatArray();
            report("global avg pool 32x2048x7x7", 32.0 * 2048 * 49, () -> F.globalAvgPool2d(x).toFloatArray());
            return null;
        });
    }

    @Test
    void convThroughput() {
        System.out.printf(Locale.ROOT, "[conv] %d threads, NCHW FP32, through the CPU backend%n", Cpu.threads());
        // ResNet-50 at batch 1 and 32: stem, a 3x3 of each stage, 1x1 expansions
        int[][] convs = { { 1, 3, 224, 64, 7, 2, 3 }, { 1, 64, 56, 64, 3, 1, 1 }, { 1, 128, 28, 128, 3, 1, 1 }, { 1, 256, 14, 256, 3, 1, 1 }, { 1, 512, 7, 512, 3, 1, 1 },
                { 1, 64, 56, 256, 1, 1, 0 }, { 1, 1024, 14, 256, 1, 1, 0 }, { 32, 64, 56, 64, 3, 1, 1 }, { 32, 256, 14, 256, 3, 1, 1 }, { 32, 1024, 14, 256, 1, 1, 0 } };
        Engine.with(Backend.CPU, () -> {
            for (int[] v : convs) {
                int n = v[0], cin = v[1], hw = v[2], cout = v[3], r = v[4], stride = v[5], pad = v[6];
                int out = (hw + 2 * pad - r) / stride + 1;
                Tensor x = Tensor.uniform(1, -1, 1, n, cin, hw, hw);
                Tensor w = Tensor.persistent(Tensor.uniform(2, -0.1f, 0.1f, cout, cin, r, r));
                double flops = 2.0 * n * cout * out * out * cin * r * r;
                report(String.format(Locale.ROOT, "conv n%d %d->%d %dx%d k%d s%d", n, cin, cout, hw, hw, r, stride), flops, () -> F.conv2d(x, w, null, stride, pad, 1, 1).toFloatArray());
            }
            return null;
        });
    }

    private static void report(String what, double flops, Runnable call) {
        call.run();
        call.run();
        long[] t = new long[RUNS];
        for (int i = 0; i < RUNS; i++) {
            long s = System.nanoTime();
            call.run();
            t[i] = System.nanoTime() - s;
        }
        java.util.Arrays.sort(t);
        double ms = t[RUNS / 2] / 1e6;
        System.out.printf(Locale.ROOT, "  %-34s median %9.3f ms  %8.1f GFLOP/s%n", what, ms, flops / (ms * 1e6));
    }

    private static MemorySegment random(Arena arena, long count, long seed) {
        MemorySegment s = arena.allocate(count * 4, 64);
        java.util.Random r = new java.util.Random(seed);
        for (long i = 0; i < count; i++) {
            s.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat() * 2 - 1);
        }
        return s;
    }
}
