package ai.beestream.jitnn;

import static ai.beestream.jitnn.Checks.close;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Supplier;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.Test;

/** The CPU backend's training kernels (convolution and BatchNorm gradients) against the reference. */
class CpuTrainingTest {

    private static void same(String what, Supplier<float[]> f, double rtol, double atol) {
        float[] expected = Engine.with(Backend.REFERENCE, f::get);
        float[] actual = Engine.with(Backend.CPU, f::get);
        close(expected, actual, rtol, atol, what);
    }

    private static Tensor u(long seed, int... dims) {
        return Tensor.uniform(seed, -1, 1, dims);
    }

    private static long fallbacks(String op) {
        long[] v = Engine.cpuStats().get(op);
        return v == null ? 0 : v[2];
    }

    /** Both gradients of sum(conv(x, w) * probe), concatenated. */
    private static float[] convGrads(int n, int cin, int hw, int cout, int r, int stride, int pad, int dil, int groups, long seed) {
        Tensor x = u(seed, n, cin, hw, hw).requiresGrad(true);
        Tensor w = u(seed + 1, cout, cin / groups, r, r).mul(0.2f).requiresGrad(true);
        Tensor y = F.conv2d(x, w, null, stride, pad, dil, groups);
        y.mul(u(seed + 2, y.shape().dims())).sum().backward();
        float[] gx = x.grad().toFloatArray();
        float[] gw = w.grad().toFloatArray();
        float[] all = java.util.Arrays.copyOf(gx, gx.length + gw.length);
        System.arraycopy(gw, 0, all, gx.length, gw.length);
        return all;
    }

    @Test
    void convolutionGradients() {
        long data = fallbacks("conv2d_bwd_data");
        long weight = fallbacks("conv2d_bwd_weight");
        same("3x3 pad 1, batch 3", () -> convGrads(3, 8, 11, 12, 3, 1, 1, 1, 1, 10), 1e-4, 1e-4);
        same("3x3 stride 2, batch 2", () -> convGrads(2, 6, 13, 10, 3, 2, 1, 1, 1, 20), 1e-4, 1e-4);
        same("1x1, batch 4", () -> convGrads(4, 16, 7, 24, 1, 1, 0, 1, 1, 30), 1e-4, 1e-4);
        same("7x7 stride 2 pad 3", () -> convGrads(2, 3, 20, 8, 7, 2, 3, 1, 1, 40), 1e-4, 1e-4);
        same("dilation 2", () -> convGrads(2, 4, 12, 6, 3, 1, 2, 2, 1, 50), 1e-4, 1e-4);
        same("grouped", () -> convGrads(2, 8, 9, 12, 3, 1, 1, 1, 4, 60), 1e-4, 1e-4);
        same("one image, 64 channels", () -> convGrads(1, 64, 14, 64, 3, 1, 1, 1, 1, 70), 1e-4, 1e-4);
        same("many small images (weight gradient over grouped images)", () -> convGrads(40, 16, 5, 16, 3, 1, 1, 1, 1, 80), 1e-4, 1e-4);
        assertEquals(data, fallbacks("conv2d_bwd_data"), "input gradients ran on the CPU");
        assertEquals(weight, fallbacks("conv2d_bwd_weight"), "weight gradients ran on the CPU");
    }

    @Test
    void batchNormTraining() {
        long before = fallbacks("batch_norm") + fallbacks("batch_norm_bwd");
        same("batch norm forward, gradients and running statistics", () -> {
            Tensor x = u(90, 4, 6, 9, 7).mul(3f).add(Tensor.scalar(0.5f)).requiresGrad(true);
            Tensor gamma = u(91, 6).requiresGrad(true);
            Tensor beta = u(92, 6).requiresGrad(true);
            Tensor rm = Tensor.persistent(u(93, 6));
            Tensor rv = Tensor.persistent(u(94, 6).abs().add(Tensor.scalar(0.5f)));
            Tensor y = F.batchNormRelu(x, rm, rv, gamma, beta, true, 0.1f, 1e-5f);
            y.mul(u(95, 4, 6, 9, 7)).sum().backward();
            float[][] parts = { y.toFloatArray(), x.grad().toFloatArray(), gamma.grad().toFloatArray(), beta.grad().toFloatArray(), rm.toFloatArray(), rv.toFloatArray() };
            int len = 0;
            for (float[] p : parts) {
                len += p.length;
            }
            float[] all = new float[len];
            int at = 0;
            for (float[] p : parts) {
                System.arraycopy(p, 0, all, at, p.length);
                at += p.length;
            }
            return all;
        }, 1e-4, 1e-5);
        assertEquals(before, fallbacks("batch_norm") + fallbacks("batch_norm_bwd"), "batch norm ran on the CPU");
    }

    private static float[] concat(Tensor... ts) {
        int len = 0;
        float[][] parts = new float[ts.length][];
        for (int i = 0; i < ts.length; i++) {
            parts[i] = ts[i].toFloatArray();
            len += parts[i].length;
        }
        float[] all = new float[len];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, all, at, p.length);
            at += p.length;
        }
        return all;
    }

    @Test
    void attentionGradients() {
        long before = fallbacks("flash_attention_bwd");
        same("sdpa [B, H, T, d] gradients", () -> {
            Tensor q = u(100, 2, 3, 20, 16).requiresGrad(true);
            Tensor k = u(101, 2, 3, 17, 16).requiresGrad(true);
            Tensor v = u(102, 2, 3, 17, 16).requiresGrad(true);
            F.scaledDotProductAttention(q, k, v, null, false, -1).mul(u(103, 2, 3, 20, 16)).sum().backward();
            return concat(q.grad(), k.grad(), v.grad());
        }, 1e-4, 1e-5);
        same("sdpa causal with a mask, gradients", () -> {
            Tensor q = u(104, 1, 2, 19, 8).requiresGrad(true);
            Tensor k = u(105, 1, 2, 19, 8).requiresGrad(true);
            Tensor v = u(106, 1, 2, 19, 8).requiresGrad(true);
            F.scaledDotProductAttention(q, k, v, u(107, 1, 1, 1, 19), true, 0.3f).mul(u(108, 1, 2, 19, 8)).sum().backward();
            return concat(q.grad(), k.grad(), v.grad());
        }, 1e-4, 1e-5);
        same("multi-head self-attention gradients (packed q, k, v)", () -> {
            Tensor x = u(109, 2, 13, 32).requiresGrad(true);
            Tensor wqkv = u(110, 96, 32).mul(0.2f).requiresGrad(true);
            Tensor wout = u(111, 32, 32).mul(0.2f).requiresGrad(true);
            F.multiHeadAttention(x, wqkv, null, wout, null, 4, null, true).mul(u(112, 2, 13, 32)).sum().backward();
            return concat(x.grad(), wqkv.grad(), wout.grad());
        }, 1e-4, 1e-5);
        assertEquals(before, fallbacks("flash_attention_bwd"), "attention gradients ran on the CPU");
    }

    @Test
    void normEmbeddingAndLossGradients() {
        long before = fallbacks("row_norm_bwd") + fallbacks("embedding_bwd") + fallbacks("cross_entropy") + fallbacks("cross_entropy_bwd");
        same("layer norm gradients", () -> {
            Tensor x = u(120, 9, 70).mul(2f).requiresGrad(true);
            Tensor w = u(121, 70).requiresGrad(true);
            Tensor b = u(122, 70).requiresGrad(true);
            F.layerNorm(x, w, b, 1e-5f).mul(u(123, 9, 70)).sum().backward();
            return concat(x.grad(), w.grad(), b.grad());
        }, 1e-4, 1e-5);
        same("rms norm gradients", () -> {
            Tensor x = u(124, 9, 70).mul(2f).requiresGrad(true);
            Tensor w = u(125, 70).requiresGrad(true);
            F.rmsNorm(x, w, 1e-6f).mul(u(126, 9, 70)).sum().backward();
            return concat(x.grad(), w.grad());
        }, 1e-4, 1e-5);
        same("embedding gradient (repeated and unused rows)", () -> {
            Tensor w = u(127, 12, 33).requiresGrad(true);
            F.embedding(Tensor.of(new int[] { 3, 0, 7, 3, 9, 3 }, 6), w).mul(u(128, 6, 33)).sum().backward();
            return w.grad().toFloatArray();
        }, 1e-5, 1e-6);
        same("cross entropy and its gradient", () -> {
            Tensor x = u(129, 7, 40).mul(3f).requiresGrad(true);
            Tensor loss = F.crossEntropy(x, Tensor.of(new int[] { 1, 39, 0, 5, 5, 12, 20 }, 7));
            loss.backward();
            return concat(loss, x.grad());
        }, 1e-5, 1e-6);
        assertEquals(before, fallbacks("row_norm_bwd") + fallbacks("embedding_bwd") + fallbacks("cross_entropy") + fallbacks("cross_entropy_bwd"), "ran on the CPU");
    }
}
