package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.close;
import static org.beehive.jitnn.Checks.seq;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.function.Supplier;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Every wave 1 operator on the TornadoVM device against the host reference. Run with
 * {@code -Paccel-tests}; the device is {@code -Djitnn.test.device=<driver>:<device>}.
 */
class OpsAccelTest {

    private static final double RTOL = 1e-5;
    private static final double ATOL = 1e-6;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    static float[] on(Backend backend, Supplier<Tensor> f) {
        return Engine.with(backend, () -> f.get().toFloatArray());
    }

    static void same(String what, Supplier<Tensor> f, double rtol, double atol) {
        float[] expected = on(Backend.REFERENCE, f);
        float[] actual = on(Backend.TORNADO, f);
        close(expected, actual, rtol, atol, what);
    }

    static void same(String what, Supplier<Tensor> f) {
        same(what, f, RTOL, ATOL);
    }

    static Tensor t(int n, float scale, float offset, int... dims) {
        return Tensor.of(seq(n, scale, offset), dims);
    }

    /** Shapes large enough for the 128 x 128 GEMM and batched convolutions (ResNet-50 and BERT-base inference sizes). */
    @Test
    void largeShapes() {
        Supplier<Tensor> a = () -> Tensor.uniform(11, -1, 1, 1100, 700);
        Supplier<Tensor> b = () -> Tensor.uniform(12, -1, 1, 700, 1050);
        same("gemm128", () -> a.get().matmul(b.get()), 1e-5, 2e-5);
        same("gemm128 A^T", () -> Tensor.uniform(13, -1, 1, 700, 1100).matmul(Tensor.uniform(22, -1, 1, 700, 1030), true, false), 1e-5, 2e-5);
        same("gemm128 B^T", () -> a.get().matmul(Tensor.uniform(14, -1, 1, 1040, 700), false, true), 1e-5, 2e-5);
        same("linear 1024x768 -> 1024", () -> F.linear(Tensor.uniform(15, -1, 1, 1024, 768), Tensor.uniform(16, -0.05f, 0.05f, 1024, 768), Tensor.uniform(17, -1, 1, 1024)), 1e-5, 2e-5);
        same("conv 8x64x56x56 3x3", () -> F.conv2d(Tensor.uniform(18, -1, 1, 8, 64, 56, 56), Tensor.uniform(19, -0.05f, 0.05f, 64, 64, 3, 3), null, 1, 1, 1, 1), 1e-5, 2e-5);
        // few output tiles: split-K, in the GEMM and in the convolution
        same("split-K linear 128x768 -> 768", () -> F.linear(Tensor.uniform(23, -1, 1, 128, 768), Tensor.uniform(24, -0.05f, 0.05f, 768, 768), Tensor.uniform(25, -1, 1, 768)), 1e-5, 2e-5);
        same("split-K matmul A^T B^T", () -> Tensor.uniform(26, -1, 1, 1000, 70).matmul(Tensor.uniform(27, -1, 1, 90, 1000), true, true), 1e-5, 2e-5);
        same("split-K conv 1x512x7x7 3x3 + bias + ReLU", () -> F.conv2dRelu(Tensor.uniform(28, -1, 1, 1, 512, 7, 7), Tensor.uniform(29, -0.02f, 0.02f, 512, 512, 3, 3), Tensor.uniform(30, -1, 1, 512), 1, 1, 1, 1), 1e-5, 2e-5);
        same("split-K grouped conv", () -> F.conv2d(Tensor.uniform(31, -1, 1, 2, 256, 5, 5), Tensor.uniform(32, -0.05f, 0.05f, 128, 64, 3, 3), null, 1, 1, 1, 4), 1e-5, 2e-5);
        same("conv 8x256x14x14 1x1 stride 2", () -> F.conv2d(Tensor.uniform(20, -1, 1, 8, 256, 14, 14), Tensor.uniform(21, -0.05f, 0.05f, 512, 256, 1, 1), null, 2, 0, 1, 1), 1e-5, 2e-5);
    }

    /** Q8_0 (GGUF blocks): device against the host reference, and the quantisation error against FP32. */
    @Test
    void q8() {
        Supplier<Tensor> x = () -> Tensor.uniform(41, -1, 1, 6, 256);
        Supplier<Tensor> w = () -> Tensor.uniform(42, -0.5f, 0.5f, 300, 256);
        // one quantisation step (max |x| / 127 = 1/127) where rounding a half differs
        same("q8 round trip", () -> F.dequantize(F.quantizeQ8(x.get())), 0, 1.0 / 127 + 1e-6);
        same("q8 linear", () -> F.linearQ8(x.get(), F.quantizeQ8(w.get()), Tensor.uniform(43, -1, 1, 300)), 1e-3, 1e-3);
        same("q8 linear batched", () -> F.linearQ8(Tensor.uniform(44, -1, 1, 2, 3, 256), F.quantizeQ8(w.get()), null), 1e-3, 1e-3);
        float[] exact = on(Backend.TORNADO, () -> F.linear(x.get(), w.get(), null));
        float[] quantised = on(Backend.TORNADO, () -> F.linearQ8(x.get(), F.quantizeQ8(w.get()), null));
        double err = 0;
        double norm = 0;
        for (int i = 0; i < exact.length; i++) {
            err += (exact[i] - quantised[i]) * (double) (exact[i] - quantised[i]);
            norm += exact[i] * (double) exact[i];
        }
        assertTrue(Math.sqrt(err / norm) < 0.01, "Q8_0 linear relative error " + Math.sqrt(err / norm));
    }

    @Test
    void binary() {
        same("add", () -> t(1003, 2, 0, 1003).add(t(1003, 1, 0.5f, 1003)));
        same("mul row broadcast", () -> t(2006, 2, 0, 2, 1003).mul(t(1003, 1, 0.5f, 1003)));
        same("div col broadcast", () -> t(2006, 2, 0, 2, 1003).div(t(2, 1, 3, 2, 1)));
        same("sub scalar", () -> t(777, 3, 1, 777).sub(Tensor.scalar(0.25f)));
        same("3-d broadcast", () -> t(60, 1, 0, 3, 4, 5).add(t(4, 1, 0, 4, 1).reshape(1, 4, 1)).mul(t(5, 1, 2, 5)));
        same("maximum/minimum", () -> t(999, 2, 0, 999).maximum(t(999, 1.5f, 0.2f, 999)).minimum(Tensor.scalar(0.8f)));
        same("comparisons", () -> t(500, 1, 0, 500).gt(t(500, 1, 0.1f, 500)).add(t(500, 1, 0, 500).eq(t(500, 1, 0, 500))));
    }

    @Test
    void unary() {
        Supplier<Tensor> x = () -> t(1029, 2, 0, 1029);
        Supplier<Tensor> pos = () -> t(1029, 1, 2, 1029);
        same("neg", () -> x.get().neg());
        same("exp", () -> x.get().exp(), 2e-6, 1e-6);
        same("log", () -> pos.get().log(), 2e-6, 1e-6);
        same("sqrt", () -> pos.get().sqrt(), 2e-6, 1e-6);
        same("abs", () -> x.get().abs());
        same("relu", () -> x.get().relu());
        same("affine", () -> x.get().mul(1.5f).add(-0.25f));
        same("pow", () -> pos.get().pow(1.7f), 1e-5, 1e-6);
        same("clamp", () -> x.get().clamp(-0.7f, 0.9f));
        same("reciprocal", () -> pos.get().reciprocal(), 2e-6, 1e-6);
        same("square", () -> x.get().square());
    }

    @Test
    void reductions() {
        same("sum axis1 group", () -> t(3000, 1, 0, 3, 1000).sum(1, false), 1e-5, 1e-4);
        same("sum axis0 group", () -> t(3000, 1, 0, 1000, 3).sum(0, false), 1e-5, 1e-4);
        same("mean axis1 thread", () -> t(21000, 1, 0, 300, 70).mean(1, false), 1e-5, 1e-5);
        same("sum all", () -> t(100_003, 1, 0.01f, 100_003).sum(), 1e-4, 1e-2);
        same("max group", () -> t(4000, 1, 0, 4, 1000).max(1, false));
        same("min thread", () -> t(2000, 1, 0, 200, 10).min(1, true));
        same("argmax", () -> t(5000, 1, 0, 50, 100).argmax(1));
    }

    @Test
    void matmul() {
        for (boolean ta : new boolean[] { false, true }) {
            for (boolean tb : new boolean[] { false, true }) {
                int m = 37;
                int n = 53;
                int k = 29;
                int[] da = ta ? new int[] { k, m } : new int[] { m, k };
                int[] db = tb ? new int[] { n, k } : new int[] { k, n };
                same("gemm tA=" + ta + " tB=" + tb, () -> t(m * k, 1, 0, da).matmul(t(k * n, 1, 0.1f, db), ta, tb), 1e-5, 1e-4);
            }
        }
        same("gemm 128", () -> t(128 * 128, 1, 0, 128, 128).matmul(t(128 * 128, 0.5f, 0, 128, 128)), 1e-4, 1e-3);
        same("permute", () -> t(2 * 3 * 50, 1, 0, 2, 3, 50).permute(2, 0, 1));
        same("transpose", () -> t(33 * 65, 1, 0, 33, 65).t());
        same("batched transpose", () -> t(3 * 17 * 40, 1, 0, 3, 17, 40).transpose(1, 2));
        same("permute last two of 4-d", () -> t(2 * 3 * 5 * 7, 1, 0, 2, 3, 5, 7).permute(0, 1, 3, 2));
    }

    @Test
    void softmaxAndLoss() {
        same("softmax long rows", () -> t(5000, 3, 0, 5, 1000).softmax(), 1e-5, 1e-7);
        same("softmax uncached rows", () -> t(3 * 5000, 3, 0, 3, 5000).softmax(), 1e-5, 1e-7);
        same("logSoftmax cached 4096", () -> t(2 * 4096, 3, 0, 2, 4096).logSoftmax(), 1e-5, 1e-5);
        same("logSoftmax short rows", () -> t(640, 3, 0, 64, 10).logSoftmax(), 1e-5, 1e-5);
        int[] labels = new int[64];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = (i * 7) % 10;
        }
        same("crossEntropy", () -> F.crossEntropy(t(640, 3, 0, 64, 10), Tensor.of(labels, 64)), 1e-5, 1e-6);
    }

    @Test
    void philoxMatchesHost() {
        float[] host = on(Backend.REFERENCE, () -> Tensor.uniform(7, -2, 2, 4099));
        float[] device = on(Backend.TORNADO, () -> Tensor.uniform(7, -2, 2, 4099));
        assertArrayEquals(host, device, "uniform is bit-exact");
        same("normal", () -> Tensor.normal(7, 1, 2, 4099), 1e-5, 1e-5);
        float[] seeded = on(Backend.TORNADO, () -> Tensor.uniform(Tensor.seed(7), -2, 2, 4099));
        assertArrayEquals(host, seeded, "uniform with a seed tensor is the same stream");
        same("normal, seed tensor", () -> Tensor.normal(Tensor.seed(7), 1, 2, 4099), 1e-5, 1e-5);
    }

    @Test
    void backwardOnDevice() {
        int[] labels = { 1, 0, 2, 1, 2 };
        Supplier<float[]> grads = () -> {
            Tensor x = t(5 * 8, 1, 0, 5, 8);
            Tensor w1 = t(16 * 8, 0.5f, 0, 16, 8).requiresGrad(true);
            Tensor b1 = t(16, 0.1f, 0, 16).requiresGrad(true);
            Tensor w2 = t(3 * 16, 0.5f, 0.1f, 3, 16).requiresGrad(true);
            Tensor loss = F.crossEntropy(F.linear(F.linear(x, w1, b1).relu(), w2, null), Tensor.of(labels, 5));
            loss.backward();
            float[] a = w1.grad().toFloatArray();
            float[] b = b1.grad().toFloatArray();
            float[] c = w2.grad().toFloatArray();
            float[] all = new float[a.length + b.length + c.length + 1];
            System.arraycopy(a, 0, all, 0, a.length);
            System.arraycopy(b, 0, all, a.length, b.length);
            System.arraycopy(c, 0, all, a.length + b.length, c.length);
            all[all.length - 1] = loss.item();
            return all;
        };
        float[] expected = Engine.with(Backend.REFERENCE, grads::get);
        float[] actual = Engine.with(Backend.TORNADO, grads::get);
        close(expected, actual, 1e-4, 1e-6, "mlp gradients");
    }

    @Test
    void dropoutMasksMatchAndScale() {
        Supplier<Tensor> drop = () -> F.dropout(t(10_000, 1, 2, 10_000), 0.3f, Tensor.of(new int[] { 1234, 77 }, 2), 5);
        float[] host = on(Backend.REFERENCE, drop);
        float[] device = on(Backend.TORNADO, drop);
        close(host, device, 1e-6, 1e-7, "same mask on host and device");
        int zeros = 0;
        for (float v : device) {
            zeros += v == 0 ? 1 : 0;
        }
        org.junit.jupiter.api.Assertions.assertEquals(0.3, zeros / 10_000.0, 0.02, "drop rate");
        float[] grad = Engine.with(Backend.TORNADO, () -> {
            Tensor x = t(10_000, 1, 2, 10_000).requiresGrad(true);
            F.dropout(x, 0.3f, Tensor.of(new int[] { 1234, 77 }, 2), 5).sum().backward();
            return x.grad().toFloatArray();
        });
        for (int i = 0; i < grad.length; i++) {
            org.junit.jupiter.api.Assertions.assertEquals(device[i] == 0 ? 0 : 1 / 0.7f, grad[i], 1e-5, "gradient uses the forward mask at " + i);
        }
    }

    @Test
    void topkIndicesMatchReference() {
        Supplier<int[]> f = () -> Tensor.uniform(5, -1.0f, 1.0f, 3, 700).topk(10)[1].toIntArray();
        int[] expected = Engine.with(Backend.REFERENCE, f::get);
        int[] actual = Engine.with(Backend.TORNADO, f::get);
        assertArrayEquals(expected, actual, "top-k indices");
    }

    @Test
    void multinomialMatchesReferenceAndDistribution() {
        int rows = 20000;
        float[] w = new float[rows * 5];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(new float[] { 0.1f, 0.0f, 0.2f, 0.3f, 0.4f }, 0, w, r * 5, 5);
        }
        Supplier<int[]> f = () -> F.multinomial(Tensor.of(w, rows, 5), Tensor.of(new int[] { 1234, 99 }, 2), 7).toIntArray();
        int[] expected = Engine.with(Backend.REFERENCE, f::get);
        int[] actual = Engine.with(Backend.TORNADO, f::get);
        int differ = 0;
        int[] counts = new int[5];
        for (int r = 0; r < rows; r++) {
            differ += expected[r] != actual[r] ? 1 : 0;
            counts[actual[r]]++;
        }
        // the device sums in float and the reference in double: draws may differ only at a boundary
        org.junit.jupiter.api.Assertions.assertTrue(differ <= 2, differ + " draws differ from the reference");
        org.junit.jupiter.api.Assertions.assertEquals(0, counts[1], "zero-weight index drawn");
        float[] p = { 0.1f, 0.0f, 0.2f, 0.3f, 0.4f };
        for (int c = 0; c < 5; c++) {
            org.junit.jupiter.api.Assertions.assertEquals(p[c], counts[c] / (float) rows, 0.012f, "frequency of " + c);
        }
    }

    @Test
    void castsMatchReferenceBitForBit() {
        float[] v = { 70000f, -65520f, 65519f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, 3.4e38f, -1e-40f, 1e-8f, 0.1f, -0.0f, 1 + 0x1p-8f, 1 + 0x3p-8f };
        for (DType d : new DType[] { DType.F16, DType.BF16 }) {
            Supplier<float[]> f = () -> Tensor.of(v, v.length).to(d).toFloat().toFloatArray();
            float[] expected = Engine.with(Backend.REFERENCE, f::get);
            float[] actual = Engine.with(Backend.TORNADO, f::get);
            for (int i = 0; i < v.length; i++) {
                org.junit.jupiter.api.Assertions.assertEquals(Float.floatToIntBits(expected[i]), Float.floatToIntBits(actual[i]), d + " cast of " + v[i] + ": " + expected[i] + " vs " + actual[i]);
            }
        }
    }
}
