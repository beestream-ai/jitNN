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

import static ai.beestream.jitnn.OpsAccelTest.same;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import ai.beestream.jitnn.engine.Native;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The operators that call vendor libraries (cuBLAS, cuDNN, cuFFT, cuSPARSE; MLX on Metal) against
 * the host reference, with forward and backward passes. Runs only with the libraries requested:
 * {@code -Paccel-tests -Djitnn.test.extra=-Djitnn.native=auto}. Shapes the libraries cannot take
 * (grouped and dilated convolutions, transposed MLX operands) check the fallback to jitNN kernels.
 */
class NativeAccelTest {

    @BeforeAll
    static void requested() {
        Assumptions.assumeTrue(Native.requested(), "vendor libraries not requested (-Djitnn.native)");
    }

    @AfterAll
    static void release() {
        Map<Native.Lib, Long> tasks = Native.emittedTasks();
        System.out.println("[native] library tasks compiled: " + tasks);
        JitnnRuntime.reset();
    }

    @Test
    void matmuls() {
        int m = 300, k = 200, n = 150;
        same("A B", () -> Tensor.uniform(1, -1, 1, m, k).matmul(Tensor.uniform(2, -1, 1, k, n)), 1e-4, 1e-4);
        same("A^T B", () -> Tensor.uniform(3, -1, 1, k, m).matmul(Tensor.uniform(4, -1, 1, k, n), true, false), 1e-4, 1e-4);
        same("A B^T", () -> Tensor.uniform(5, -1, 1, m, k).matmul(Tensor.uniform(6, -1, 1, n, k), false, true), 1e-4, 1e-4);
        same("A^T B^T", () -> Tensor.uniform(7, -1, 1, k, m).matmul(Tensor.uniform(8, -1, 1, n, k), true, true), 1e-4, 1e-4);
        // host inputs only: the plan is a single library task, with no launch grid at all
        same("library task only", () -> Tensor.of(Checks.seq(48 * 32, 1, 0), 48, 32).matmul(Tensor.of(Checks.seq(32 * 40, 0.5f, 0.1f), 32, 40)), 1e-4, 1e-4);
        same("vector", () -> Tensor.uniform(9, -1, 1, 1, k).matmul(Tensor.uniform(10, -1, 1, k, n)), 1e-4, 1e-4);
        same("batched", () -> Tensor.uniform(11, -1, 1, 6, m, k).matmul(Tensor.uniform(12, -1, 1, 6, k, n)), 1e-4, 1e-4);
        same("batched, shared B", () -> Tensor.uniform(13, -1, 1, 6, m, k).matmul(Tensor.uniform(14, -1, 1, k, n)), 1e-4, 1e-4);
        same("FP16 operands", () -> Tensor.uniform(15, -1, 1, 256, 128).half().matmul(Tensor.uniform(16, -1, 1, 128, 192).half()), 1e-3, 1e-3);
        same("matmul gradients", () -> {
            Tensor a = Tensor.uniform(17, -1, 1, 64, 96).requiresGrad(true);
            Tensor b = Tensor.uniform(18, -1, 1, 96, 80).requiresGrad(true);
            a.matmul(b).mul(Tensor.uniform(19, -1, 1, 64, 80)).sum().backward();
            return Tensor.cat(0, a.grad().reshape(-1), b.grad().reshape(-1));
        }, 1e-4, 1e-4);
    }

    @Test
    void linearLayers() {
        same("linear + bias", () -> F.linear(Tensor.uniform(20, -1, 1, 128, 96), Tensor.uniform(21, -0.1f, 0.1f, 64, 96), Tensor.uniform(22, -1, 1, 64)), 1e-4, 1e-4);
        same("linear + bias + ReLU", () -> F.linearRelu(Tensor.uniform(23, -1, 1, 128, 96), Tensor.uniform(24, -0.1f, 0.1f, 64, 96), Tensor.uniform(25, -1, 1, 64)), 1e-4, 1e-4);
        same("linear gradients", () -> {
            Tensor x = Tensor.uniform(26, -1, 1, 32, 48).requiresGrad(true);
            Tensor w = Tensor.uniform(27, -0.1f, 0.1f, 24, 48).requiresGrad(true);
            Tensor bias = Tensor.uniform(28, -1, 1, 24).requiresGrad(true);
            F.linear(x, w, bias).mul(Tensor.uniform(29, -1, 1, 32, 24)).sum().backward();
            return Tensor.cat(0, x.grad().reshape(-1), w.grad().reshape(-1), bias.grad().reshape(-1));
        }, 1e-4, 1e-4);
    }

    /**
     * cuDNN runs FP32 convolutions on TF32 tensor cores on Ampere and newer (its default math, as in
     * PyTorch); NVIDIA_TF32_OVERRIDE=0 makes them FP32, and then the FP32 tolerance holds.
     */
    private static final double CONV_TOL = "0".equals(System.getenv("NVIDIA_TF32_OVERRIDE")) ? 1e-4 : 3e-3;

    @Test
    void convolutions() {
        // N, Cin, H, Cout, K, stride, pad, dilation, groups
        int[][] cases = { { 4, 16, 14, 32, 3, 1, 1, 1, 1 }, { 4, 16, 15, 24, 3, 2, 1, 1, 1 }, { 2, 8, 9, 12, 5, 1, 2, 1, 1 }, { 8, 64, 8, 128, 1, 1, 0, 1, 1 },
                { 2, 16, 12, 16, 3, 1, 1, 1, 4 }, { 2, 8, 12, 8, 3, 1, 2, 2, 1 }, { 32, 32, 16, 32, 3, 1, 1, 1, 1 } };
        for (int[] c : cases) {
            int n = c[0], cin = c[1], h = c[2], cout = c[3], k = c[4], st = c[5], pd = c[6], dl = c[7], gr = c[8];
            String what = "conv " + n + "x" + cin + "x" + h + "x" + h + " -> " + cout + " k" + k + " s" + st + " p" + pd + " d" + dl + " g" + gr;
            same(what, () -> {
                Tensor x = Tensor.uniform(30, -1, 1, n, cin, h, h).requiresGrad(true);
                Tensor w = Tensor.uniform(31, -0.1f, 0.1f, cout, cin / gr, k, k).requiresGrad(true);
                Tensor bias = Tensor.uniform(32, -1, 1, cout).requiresGrad(true);
                Tensor y = F.conv2d(x, w, bias, st, pd, dl, gr);
                y.mul(Tensor.uniform(33, -1, 1, y.shape().dims())).sum().backward();
                return Tensor.cat(0, y.reshape(-1), x.grad().reshape(-1), w.grad().reshape(-1), bias.grad().reshape(-1));
            }, CONV_TOL, CONV_TOL);
            same(what + " + ReLU", () -> F.conv2dRelu(Tensor.uniform(34, -1, 1, n, cin, h, h), Tensor.uniform(35, -0.1f, 0.1f, cout, cin / gr, k, k), Tensor.uniform(36, -1, 1, cout), st, pd,
                    dl, gr), CONV_TOL, CONV_TOL);
        }
    }

    /**
     * Many convolutions of different shapes in one trace: more tasks than one TornadoVM task graph
     * holds, and cuDNN workspaces that grow from one convolution to the next.
     */
    @Test
    void convolutionChain() {
        same("26 convolutions", () -> {
            Tensor x = Tensor.uniform(70, -1, 1, 8, 16, 32, 32);
            int c = 16;
            for (int i = 0; i < 26; i++) {
                int out = 16 + 8 * (i % 5);
                int k = i % 3 == 0 ? 1 : 3;
                x = F.conv2dRelu(x, Tensor.uniform(71 + i, -0.2f, 0.2f, out, c, k, k), Tensor.uniform(100 + i, -0.1f, 0.1f, out), 1, k / 2, 1, 1);
                c = out;
            }
            return x;
        }, CONV_TOL, CONV_TOL);
    }

    /** Inference attention and norms (MLX on Metal; cuDNN's FP16 attention on CUDA with -Djitnn.native.attentionFp16=true). */
    @Test
    void inferenceAttentionAndNorms() {
        double attTol = Boolean.getBoolean("jitnn.native.attentionFp16") ? 5e-3 : 1e-4;
        for (boolean causal : new boolean[] { false, true }) {
            same("attention b2 h4 t128 d64 causal " + causal, () -> Grad.inference(() -> F.scaledDotProductAttention(Tensor.uniform(80, -1, 1, 2, 4, 128, 64),
                    Tensor.uniform(81, -1, 1, 2, 4, 128, 64), Tensor.uniform(82, -1, 1, 2, 4, 128, 64), null, causal, -1)), attTol, attTol);
        }
        same("attention decode t1 s200 d128", () -> Grad.inference(() -> F.scaledDotProductAttention(Tensor.uniform(83, -1, 1, 1, 8, 1, 128),
                Tensor.uniform(84, -1, 1, 1, 8, 200, 128), Tensor.uniform(85, -1, 1, 1, 8, 200, 128), null, false, -1)), attTol, attTol);
        same("layer norm", () -> Grad.inference(() -> F.layerNorm(Tensor.uniform(86, -2, 2, 256, 384), Tensor.uniform(87, 0.5f, 1.5f, 384), Tensor.uniform(88, -1, 1, 384), 1e-5f)),
                1e-4, 1e-4);
        same("rms norm", () -> Grad.inference(() -> F.rmsNorm(Tensor.uniform(89, -2, 2, 256, 384), Tensor.uniform(90, 0.5f, 1.5f, 384), 1e-6f)), 1e-4, 1e-4);
    }

    @Test
    void softmax() {
        same("softmax 64 x 1000", () -> Tensor.uniform(40, -4, 4, 64, 1000).softmax(), 1e-5, 1e-6);
        same("softmax 512 x 37", () -> Tensor.uniform(41, -4, 4, 512, 37).softmax(), 1e-5, 1e-6);
    }

    @Test
    void ffts() {
        same("fft 32 x 256", () -> Fft.fft(Tensor.uniform(50, -1, 1, 32, 256, 2)), 1e-4, 1e-4);
        same("ifft 16 x 100", () -> Fft.ifft(Tensor.uniform(51, -1, 1, 16, 100, 2)), 1e-4, 1e-5);
        same("rfft 8 x 512", () -> Fft.rfft(Tensor.uniform(52, -1, 1, 8, 512)), 1e-4, 1e-4);
        same("fft cropped (jitNN kernel)", () -> Fft.fft(Tensor.uniform(53, -1, 1, 4, 64, 2), 48, -1, Fft.Norm.BACKWARD), 1e-4, 1e-4);
    }

    @Test
    void sparse() {
        int rows = 200, cols = 150, feat = 33;
        java.util.Random r = new java.util.Random(7);
        int nnz = 1500;
        int[] ri = new int[nnz];
        int[] ci = new int[nnz];
        for (int i = 0; i < nnz; i++) {
            ri[i] = r.nextInt(rows);
            ci[i] = r.nextInt(cols);
        }
        same("spmm", () -> {
            SparseMatrix a = SparseMatrix.fromCoo(rows, cols, ri, ci);
            return a.mm(Tensor.uniform(60, -1, 1, a.nnz()), Tensor.uniform(61, -1, 1, cols, feat));
        }, 1e-4, 1e-5);
    }

    /**
     * With {@code -Djitnn.native.unavailable=cudnn} cuDNN counts as missing: convolutions and
     * softmax fall back to jitNN's kernels (FP32 results), and no cuDNN task is compiled.
     */
    @Test
    void missingLibraryFallsBack() {
        Assumptions.assumeTrue(System.getProperty("jitnn.native.unavailable", "").contains("cudnn"), "set -Djitnn.native.unavailable=cudnn");
        same("conv without cuDNN", () -> F.conv2dRelu(Tensor.uniform(110, -1, 1, 8, 32, 16, 16), Tensor.uniform(111, -0.1f, 0.1f, 64, 32, 3, 3), Tensor.uniform(112, -1, 1, 64), 1, 1, 1, 1),
                1e-4, 1e-4);
        same("softmax without cuDNN", () -> Tensor.uniform(113, -4, 4, 64, 1000).softmax(), 1e-5, 1e-6);
        org.junit.jupiter.api.Assertions.assertNull(Native.emittedTasks().get(Native.Lib.CUDNN), "cuDNN tasks were compiled");
    }

    @Test
    void librariesWereCalled() {
        matmuls();
        Map<Native.Lib, Long> tasks = Native.emittedTasks();
        assertTrue(!tasks.isEmpty(), "no library task was compiled; the libraries were not found (see the [jitnn] native lines)");
    }
}
