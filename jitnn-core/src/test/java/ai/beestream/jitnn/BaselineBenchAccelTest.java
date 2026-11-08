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
package ai.beestream.jitnn;

import java.util.function.Supplier;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Baseline speed of the core kernels, printed rather than asserted. Inputs are persistent so
 * only kernel time and dispatch are measured. Run on demand:
 * {@code ./mvnw test -pl jitnn-core -Paccel-tests -Dtest=BaselineBenchAccelTest}.
 */
class BaselineBenchAccelTest {

    private static final int REPS = 30;

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    static Tensor persistent(int... dims) {
        return Tensor.persistent(Tensor.uniform(3, -1, 1, dims));
    }

    private static double kernelSeconds;

    /** Wall seconds per execution; also records the median device kernel time when profiling. */
    static double seconds(Supplier<Tensor> op) {
        for (int i = 0; i < 3; i++) {
            op.get();
            JitnnRuntime.sync();
        }
        long[] kernel = new long[REPS];
        long t0 = System.nanoTime();
        for (int i = 0; i < REPS; i++) {
            op.get();
            JitnnRuntime.sync();
            kernel[i] = JitnnRuntime.lastProfile()[0];
        }
        double wall = (System.nanoTime() - t0) / 1e9 / REPS;
        java.util.Arrays.sort(kernel);
        kernelSeconds = kernel[REPS / 2] / 1e9;
        return wall;
    }

    static void bandwidth(String name, double bytes, Supplier<Tensor> op) {
        double s = seconds(op);
        System.out.printf("[bench] %-28s %8.3f ms  %7.1f GB/s   kernel %8.3f ms  %7.1f GB/s%n", name, s * 1e3, bytes / s / 1e9, kernelSeconds * 1e3,
                kernelSeconds > 0 ? bytes / kernelSeconds / 1e9 : 0);
    }

    static void compute(String name, double flops, Supplier<Tensor> op) {
        double s = seconds(op);
        System.out.printf("[bench] %-28s %8.3f ms  %7.1f GFLOP/s kernel %8.3f ms  %7.1f GFLOP/s%n", name, s * 1e3, flops / s / 1e9, kernelSeconds * 1e3,
                kernelSeconds > 0 ? flops / kernelSeconds / 1e9 : 0);
    }

    @Test
    void baseline() {
        int n = 1 << 24;
        Tensor a = persistent(n);
        Tensor b = persistent(n);
        Tensor row = persistent(4096);
        Tensor m = persistent(4096, 4096);
        bandwidth("add 16M", 12.0 * n, () -> a.add(b));
        bandwidth("add broadcast 4096x4096", 8.0 * n, () -> m.add(row));
        bandwidth("exp 16M", 8.0 * n, a::exp);
        bandwidth("sum 16M", 4.0 * n, a::sum);
        bandwidth("sum axis0 4096x4096", 4.0 * n, () -> m.sum(0, false));
        bandwidth("sum axis1 4096x4096", 4.0 * n, () -> m.sum(1, false));
        bandwidth("softmax 4096x4096", 8.0 * n, m::softmax);
        bandwidth("transpose 4096x4096", 8.0 * n, m::t);
        for (int s : new int[] { 512, 1024, 2048 }) {
            Tensor x = persistent(s, s);
            Tensor y = persistent(s, s);
            compute("gemm " + s, 2.0 * s * s * s, () -> x.matmul(y));
            compute("gemm " + s + " B^T", 2.0 * s * s * s, () -> x.matmul(y, false, true));
        }
        // half-precision operands (cast inside the timed step), FP32 accumulation: tensor cores on CUDA
        for (int s : new int[] { 1024, 2048, 4096 }) {
            Tensor x = persistent(s, s);
            Tensor y = persistent(s, s);
            compute("gemm " + s + " fp16", 2.0 * s * s * s, () -> x.half().matmul(y.half()));
            compute("gemm " + s + " bf16", 2.0 * s * s * s, () -> x.bfloat16().matmul(y.bfloat16()));
            compute("gemm " + s + " fp16 B^T", 2.0 * s * s * s, () -> x.half().matmul(y.half(), false, true));
            compute("gemm " + s + " bf16 B^T", 2.0 * s * s * s, () -> x.bfloat16().matmul(y.bfloat16(), false, true));
        }
    }
}
