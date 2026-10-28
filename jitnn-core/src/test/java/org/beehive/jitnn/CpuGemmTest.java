package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.close;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Supplier;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.Test;

/** The CPU backend's GEMM-based operators (matmul, linear, conv2d) against the reference, edge tiles and epilogues included. */
class CpuGemmTest {

    private static void same(String what, Supplier<Tensor> f) {
        same(what, f, 1e-5, 2e-5);
    }

    private static void same(String what, Supplier<Tensor> f, double rtol, double atol) {
        float[] expected = Engine.with(Backend.REFERENCE, () -> f.get().toFloatArray());
        float[] actual = Engine.with(Backend.CPU, () -> f.get().toFloatArray());
        close(expected, actual, rtol, atol, what);
    }

    private static Tensor u(long seed, int... dims) {
        return Tensor.uniform(seed, -1, 1, dims);
    }

    private static long fallbacks(String op) {
        long[] v = Engine.cpuStats().get(op);
        return v == null ? 0 : v[2];
    }

    @Test
    void matmulShapesAndTransposes() {
        long before = fallbacks("matmul");
        int[][] shapes = { { 1, 1, 1 }, { 7, 13, 5 }, { 6, 32, 8 }, { 100, 37, 300 }, { 65, 129, 600 }, { 128, 768, 64 } };
        for (int[] s : shapes) {
            int m = s[0], n = s[1], k = s[2];
            same("matmul " + m + "x" + n + "x" + k, () -> u(1, m, k).matmul(u(2, k, n)));
            same("matmul A^T " + m + "x" + n + "x" + k, () -> u(3, k, m).matmul(u(4, k, n), true, false));
            same("matmul B^T " + m + "x" + n + "x" + k, () -> u(5, m, k).matmul(u(6, n, k), false, true));
            same("matmul A^T B^T " + m + "x" + n + "x" + k, () -> u(7, k, m).matmul(u(8, n, k), true, true));
        }
        assertEquals(before, fallbacks("matmul"), "every matmul ran the CPU kernel");
    }

    @Test
    void batchedMatmul() {
        same("batched 12 x [33, 64] [64, 47]", () -> u(10, 12, 33, 64).matmul(u(11, 12, 64, 47)));
        same("batched attention scores 2x4 x [17, 16] [17, 16]^T", () -> u(12, 2, 4, 17, 16).matmul(u(13, 2, 4, 17, 16), false, true));
        same("batched, shared right matrix", () -> u(14, 5, 9, 20).matmul(u(15, 20, 11)));
        same("batched, shared left matrix", () -> u(16, 9, 20).matmul(u(17, 5, 20, 11)));
    }

    @Test
    void linearEpilogues() {
        long before = fallbacks("linear");
        same("linear", () -> F.linear(u(20, 37, 129), u(21, 70, 129), null));
        same("linear + bias", () -> F.linear(u(22, 37, 129), u(23, 70, 129), u(24, 70)));
        same("linear + bias + ReLU", () -> F.linear(u(25, 64, 300), u(26, 130, 300), u(27, 130)).relu());
        same("linear, weight persistent (packed once)", () -> {
            Tensor w = Tensor.persistent(u(28, 33, 40));
            F.linear(u(29, 8, 40), w, null);
            return F.linear(u(30, 8, 40), w, null);
        });
        same("linear + GELU fused in inference", () -> Grad.inference(() -> F.linear(u(31, 19, 64), u(32, 50, 64), u(33, 50)).gelu()));
        assertEquals(before, fallbacks("linear"), "every linear ran the CPU kernel");
    }

    @Test
    void convolutions() {
        long before = fallbacks("conv2d");
        same("conv 1x1", () -> F.conv2d(u(40, 2, 16, 9, 7), u(41, 24, 16, 1, 1), null, 1, 0, 1, 1));
        same("conv 1x1 stride 2 + bias", () -> F.conv2d(u(42, 2, 16, 9, 7), u(43, 24, 16, 1, 1), u(44, 24), 2, 0, 1, 1));
        same("conv 3x3 pad 1 + bias + ReLU", () -> F.conv2dRelu(u(45, 3, 8, 13, 11), u(46, 20, 8, 3, 3), u(47, 20), 1, 1, 1, 1));
        same("conv 7x7 stride 2 pad 3", () -> F.conv2d(u(48, 2, 3, 30, 28), u(49, 16, 3, 7, 7), null, 2, 3, 1, 1));
        same("conv 3x3 dilation 2", () -> F.conv2d(u(50, 1, 6, 15, 15), u(51, 10, 6, 3, 3), null, 1, 2, 2, 1));
        same("conv grouped", () -> F.conv2d(u(52, 2, 12, 10, 10), u(53, 16, 3, 3, 3), u(54, 16), 1, 1, 1, 4));
        same("conv depthwise", () -> F.conv2d(u(55, 2, 8, 10, 10), u(56, 8, 1, 3, 3), null, 1, 1, 1, 8));
        same("conv 3x3 64 channels, deep", () -> F.conv2d(u(57, 2, 64, 14, 14), u(58, 64, 64, 3, 3), null, 1, 1, 1, 1), 1e-5, 5e-5);
        same("conv batch, dilation 2, stride 2", () -> F.conv2d(u(59, 3, 6, 17, 15), u(60, 10, 6, 3, 3), u(61, 10), 2, 2, 2, 1));
        same("conv batch of 32 x 32 images", () -> F.conv2dRelu(u(62, 4, 64, 32, 32), u(63, 64, 64, 3, 3).mul(0.1f), u(64, 64), 1, 1, 1, 1), 1e-5, 5e-5);
        same("conv batch of 48 small maps", () -> F.conv2dRelu(u(65, 48, 64, 14, 14), u(66, 64, 64, 3, 3).mul(0.1f), u(67, 64), 1, 1, 1, 1), 1e-5,
                5e-5);
        assertEquals(before, fallbacks("conv2d"), "every convolution ran the CPU kernel");
    }
}
