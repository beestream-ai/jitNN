package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.close;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Supplier;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.Test;

/**
 * The CPU backend's element-wise, row, pooling, reduction, embedding and attention kernels against
 * the reference: broadcasts, tails shorter than a vector, masks and layouts; and none of them falls
 * back to the reference.
 */
class CpuOpsTest {

    private static void same(String what, Supplier<Tensor> f) {
        same(what, f, 1e-5, 2e-6);
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
    void binaryBroadcasts() {
        long before = fallbacks("binary");
        same("add, same shape, tail", () -> u(1, 3, 37).add(u(2, 3, 37)));
        same("sub, scalar right", () -> u(3, 5, 41).sub(Tensor.scalar(0.25f)));
        same("mul, scalar left", () -> Tensor.scalar(3f).mul(u(4, 5, 41)));
        same("div, row bias [N, C] / [C]", () -> u(5, 7, 33).div(u(6, 33).add(Tensor.scalar(2f))));
        same("add, channel [N, C, H, W] + [1, C, 1, 1]", () -> u(7, 2, 5, 9, 11).add(u(8, 1, 5, 1, 1)));
        same("add, outer [N, 1] + [1, M]", () -> u(9, 13, 1).add(u(10, 1, 29)));
        same("maximum", () -> u(11, 4, 50).maximum(u(12, 4, 50)));
        same("relu then add (fused add + ReLU)", () -> Grad.inference(() -> u(13, 6, 70).add(u(14, 6, 70)).relu()));
        same("relu gradient", () -> {
            Tensor x = u(15, 4, 45).requiresGrad(true);
            x.relu().mul(u(16, 4, 45)).sum().backward();
            return x.grad();
        });
        same("pow, scalar path", () -> org.beehive.jitnn.ops.Binary.apply(org.beehive.jitnn.ops.Binary.Kind.POW, u(17, 3, 20).abs().add(Tensor.scalar(0.5f)), u(18, 3, 20)), 1e-5, 1e-6);
        assertEquals(before, fallbacks("binary"), "every broadcast ran the CPU kernel");
    }

    @Test
    void unaryOps() {
        long before = fallbacks("unary");
        same("neg", () -> u(20, 3, 37).neg());
        same("abs", () -> u(21, 3, 37).abs());
        same("relu", () -> u(22, 3, 37).relu());
        same("affine", () -> u(23, 3, 37).mul(3f));
        same("clamp", () -> u(24, 3, 37).clamp(-0.3f, 0.4f));
        same("sqrt", () -> u(25, 3, 37).abs().sqrt());
        same("exp", () -> u(26, 3, 37).mul(5f).exp(), 2e-6, 1e-6);
        same("sigmoid", () -> u(27, 3, 37).mul(8f).sigmoid(), 2e-6, 1e-6);
        same("silu", () -> u(28, 3, 37).mul(8f).silu(), 2e-6, 1e-6);
        same("tanh", () -> u(29, 3, 37).mul(4f).tanh(), 2e-6, 1e-6);
        same("leaky relu", () -> u(30, 3, 37).leakyRelu(0.1f));
        same("gelu (scalar path)", () -> u(31, 3, 37).mul(3f).gelu(), 2e-6, 1e-6);
        assertEquals(before, fallbacks("unary"), "every unary ran the CPU kernel");
    }

    @Test
    void rowsAndReductions() {
        same("softmax", () -> u(40, 9, 77).mul(4f).softmax(), 2e-6, 1e-7);
        same("log softmax", () -> u(41, 9, 77).mul(4f).logSoftmax(), 2e-6, 1e-6);
        same("layer norm", () -> F.layerNorm(u(42, 10, 99), u(43, 99), u(44, 99), 1e-5f), 1e-5, 1e-5);
        same("rms norm", () -> F.rmsNorm(u(45, 10, 99), u(46, 99), 1e-6f), 1e-5, 1e-5);
        same("sum of rows", () -> u(47, 8, 101).sum(1, false), 1e-5, 1e-5);
        same("mean over a middle axis", () -> u(48, 4, 33, 7).mean(1, false), 1e-5, 1e-6);
        same("max over the first axis", () -> u(49, 6, 5, 9).max(0, false));
        same("sum of everything", () -> u(50, 300, 70).sum(), 1e-5, 1e-4);
        assertEquals(0, fallbacks("softmax") + fallbacks("row_norm") + fallbacks("reduce"), "row kernels ran on the CPU");
    }

    @Test
    void poolingAndEmbedding() {
        same("max pool 3x3 stride 2 pad 1", () -> F.maxPool2d(u(60, 2, 3, 15, 13), 3, 2, 1));
        same("avg pool 2x2", () -> F.avgPool2d(u(61, 2, 3, 14, 12), 2, 2, 0));
        same("max pool gradient", () -> {
            Tensor x = u(62, 1, 2, 9, 9).requiresGrad(true);
            F.maxPool2d(x, 3, 2, 1).mul(u(63, 1, 2, 5, 5)).sum().backward();
            return x.grad();
        });
        same("embedding", () -> F.embedding(Tensor.of(new int[] { 3, 0, 7, 3, 9 }, 5), u(64, 10, 21)));
        assertEquals(0, fallbacks("pool2d") + fallbacks("embedding"), "pooling and embedding ran on the CPU");
    }

    @Test
    void attention() {
        long before = fallbacks("flash_attention");
        same("sdpa [B, H, T, d]", () -> F.scaledDotProductAttention(u(70, 2, 3, 70, 16), u(71, 2, 3, 50, 16), u(72, 2, 3, 50, 16), null, false, -1), 1e-5, 2e-6);
        same("sdpa causal, 130 rows (three blocks)", () -> F.scaledDotProductAttention(u(73, 1, 2, 130, 32), u(74, 1, 2, 130, 32), u(75, 1, 2, 130, 32), null, true, -1), 1e-5,
                2e-6);
        same("sdpa with an additive mask", () -> F.scaledDotProductAttention(u(76, 2, 2, 9, 8), u(77, 2, 2, 11, 8), u(78, 2, 2, 11, 8), u(79, 2, 1, 1, 11).mul(3f), false, 0.4f),
                1e-5, 2e-6);
        same("sdpa [B, T, H, d]", () -> F.scaledDotProductAttentionBTHD(u(80, 2, 17, 4, 16), u(81, 2, 19, 4, 16), u(82, 2, 19, 4, 16), null, false, -1), 1e-5, 2e-6);
        same("multi-head self-attention (packed q, k, v)", () -> F.multiHeadAttention(u(83, 2, 21, 64), u(84, 192, 64).mul(0.2f), u(85, 192), u(86, 64, 64).mul(0.2f), u(87, 64),
                4, null, true), 1e-5, 2e-6);
        assertEquals(before, fallbacks("flash_attention"), "attention ran on the CPU");
    }
}
