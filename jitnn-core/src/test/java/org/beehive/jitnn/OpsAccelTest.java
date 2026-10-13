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

    /** A model loaded again after the first was closed (its weights released) reuses its plans. */
    @Test
    void releasedWeightsReusePlans() {
        java.util.function.BiFunction<Tensor[], Tensor, Tensor> model = (w, x) -> F.linear(F.linear(x, w[0], w[1]).relu(), w[2], w[3]);
        // host weights, as a model file gives them; each "load" has different values
        java.util.function.IntFunction<Tensor[]> load = seed -> new Tensor[] { Tensor.persistent(t(64 * 48, 0.001f * seed, -0.1f, 64, 48)),
                Tensor.persistent(t(64, 0.01f, 0.001f * seed, 64)), Tensor.persistent(t(10 * 64, -0.0005f * seed, 0.1f, 10, 64)),
                Tensor.persistent(t(10, 0.1f, -0.001f * seed, 10)) };
        Supplier<Tensor> x = () -> Tensor.uniform(5, -1, 1, 8, 48);
        float[][] expected = new float[3][];
        for (int m = 0; m < 3; m++) {
            int seed = 200 + 10 * m;
            expected[m] = on(Backend.REFERENCE, () -> model.apply(load.apply(seed), x.get()));
        }
        Engine.with(Backend.TORNADO, () -> {
            Tensor[] first = load.apply(200);
            close(expected[0], model.apply(first, x.get()).toFloatArray(), RTOL, ATOL, "first model");
            long[] before = Engine.planStats();
            for (Tensor w : first) {
                w.release();
            }
            for (int m = 1; m < 3; m++) {
                Tensor[] again = load.apply(200 + 10 * m);
                close(expected[m], model.apply(again, x.get()).toFloatArray(), RTOL, ATOL, "model loaded again " + m);
                close(expected[m], model.apply(again, x.get()).toFloatArray(), RTOL, ATOL, "model loaded again " + m + ", replayed");
                for (Tensor w : again) {
                    w.release();
                }
            }
            long[] after = Engine.planStats();
            org.junit.jupiter.api.Assertions.assertEquals(before[0], after[0], "plans compiled for the reloaded models");
            org.junit.jupiter.api.Assertions.assertEquals(before[2] + 2, after[2], "plans reused");
            org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, () -> model.apply(first, x.get()).toFloatArray());
            return null;
        });
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
        // Winograd: odd output sizes, padding 0, and the input gradient through the flipped filter
        same("conv 3x3 4x24x21x23 pad 0 + bias", () -> F.conv2d(Tensor.uniform(33, -1, 1, 4, 24, 21, 23), Tensor.uniform(34, -0.1f, 0.1f, 40, 24, 3, 3), Tensor.uniform(35, -1, 1, 40), 1, 0, 1, 1), 1e-5, 2e-5);
        same("conv 3x3 input gradient", () -> {
            Tensor x = Tensor.uniform(36, -1, 1, 4, 24, 17, 15).requiresGrad(true);
            F.conv2d(x, Tensor.uniform(37, -0.1f, 0.1f, 40, 24, 3, 3), null, 1, 1, 1, 1).mul(Tensor.uniform(38, -1, 1, 4, 40, 17, 15)).sum().backward();
            return x.grad();
        }, 1e-5, 2e-5);
        // Winograd F(4x4, 3x3): whole and partial 4x4 output tiles
        same("conv 3x3 4x24x32x28 F(4x4) + bias + ReLU", () -> F.conv2dRelu(Tensor.uniform(62, -1, 1, 4, 24, 32, 28), Tensor.uniform(63, -0.1f, 0.1f, 40, 24, 3, 3), Tensor.uniform(64, -1, 1, 40), 1, 1, 1, 1), 1e-5, 2e-5);
        same("conv 3x3 F(4x4) input gradient, partial tiles", () -> {
            Tensor x = Tensor.uniform(65, -1, 1, 4, 24, 30, 26).requiresGrad(true);
            F.conv2d(x, Tensor.uniform(66, -0.1f, 0.1f, 40, 24, 3, 3), null, 1, 1, 1, 1).mul(Tensor.uniform(67, -1, 1, 4, 40, 30, 26)).sum().backward();
            return x.grad();
        }, 1e-5, 2e-5);
        same("conv 3x3 filter gradient, sliced tiles", () -> {
            Tensor w = Tensor.uniform(46, -0.1f, 0.1f, 40, 24, 3, 3).requiresGrad(true);
            F.conv2d(Tensor.uniform(47, -1, 1, 8, 24, 31, 33), w, null, 1, 1, 1, 1).mul(Tensor.uniform(48, -1, 1, 8, 40, 31, 33)).sum().backward();
            return w.grad();
        }, 1e-4, 1e-4);
        // strided input gradients: W^T dy as one GEMM gathered into dx (one group), or one implicit GEMM per stride phase
        same("conv 3x3 stride 2 input gradient", () -> {
            Tensor x = Tensor.uniform(50, -1, 1, 4, 16, 13, 11).requiresGrad(true);
            Tensor y = F.conv2d(x, Tensor.uniform(51, -0.1f, 0.1f, 24, 16, 3, 3), null, 2, 1, 1, 1);
            y.mul(Tensor.uniform(52, -1, 1, y.shape().dims())).sum().backward();
            return x.grad();
        }, 1e-5, 2e-5);
        same("conv 1x1 stride 2 input gradient", () -> {
            Tensor x = Tensor.uniform(53, -1, 1, 2, 16, 14, 14).requiresGrad(true);
            Tensor y = F.conv2d(x, Tensor.uniform(54, -0.1f, 0.1f, 32, 16, 1, 1), null, 2, 0, 1, 1);
            y.mul(Tensor.uniform(55, -1, 1, y.shape().dims())).sum().backward();
            return x.grad();
        }, 1e-5, 2e-5);
        same("conv 5x5 stride 3 input gradient", () -> {
            Tensor x = Tensor.uniform(56, -1, 1, 2, 8, 15, 16).requiresGrad(true);
            Tensor y = F.conv2d(x, Tensor.uniform(57, -0.1f, 0.1f, 12, 8, 5, 5), null, 3, 2, 1, 1);
            y.mul(Tensor.uniform(58, -1, 1, y.shape().dims())).sum().backward();
            return x.grad();
        }, 1e-5, 2e-5);
        same("grouped conv 3x3 stride 2 input gradient", () -> {
            Tensor x = Tensor.uniform(59, -1, 1, 2, 16, 13, 11).requiresGrad(true);
            Tensor y = F.conv2d(x, Tensor.uniform(60, -0.1f, 0.1f, 24, 8, 3, 3), null, 2, 1, 1, 2);
            y.mul(Tensor.uniform(61, -1, 1, y.shape().dims())).sum().backward();
            return x.grad();
        }, 1e-5, 2e-5);
        same("conv 8x64x24x24 1x1 + bias + ReLU", () -> F.conv2dRelu(Tensor.uniform(39, -1, 1, 8, 64, 24, 24), Tensor.uniform(40, -0.1f, 0.1f, 96, 64, 1, 1), Tensor.uniform(43, -1, 1, 96), 1, 0, 1, 1), 1e-5, 2e-5);
        same("conv 8x64x24x24 1x1", () -> F.conv2d(Tensor.uniform(44, -1, 1, 8, 64, 24, 24), Tensor.uniform(45, -0.1f, 0.1f, 96, 64, 1, 1), null, 1, 0, 1, 1), 1e-5, 2e-5);
        same("conv 8x256x14x14 1x1 stride 2", () -> F.conv2d(Tensor.uniform(20, -1, 1, 8, 256, 14, 14), Tensor.uniform(21, -0.05f, 0.05f, 512, 256, 1, 1), null, 2, 0, 1, 1), 1e-5, 2e-5);
    }

    /** Attention at head dimensions 33 to 64 (the tiled kernels): masks, causal, T != S, both layouts, the packed projection, gradients. */
    @Test
    void attentionHead64() {
        float[] pad = new float[2 * 150];
        for (int i = 0; i < pad.length; i++) {
            pad[i] = i % 150 >= (i < 150 ? 130 : 110) ? -1e9f : 0.0f;
        }
        Tensor keyMask = Tensor.of(pad, 2, 1, 1, 150);
        same("sdpa d64 T100 S150 + key mask", () -> F.scaledDotProductAttention(Tensor.uniform(70, -1, 1, 2, 3, 100, 64), Tensor.uniform(71, -1, 1, 2, 3, 150, 64),
                Tensor.uniform(72, -1, 1, 2, 3, 150, 64), keyMask, false, -1.0f), 1e-5, 2e-5);
        same("sdpa d64 T130 causal", () -> F.scaledDotProductAttention(Tensor.uniform(73, -1, 1, 2, 2, 130, 64), Tensor.uniform(74, -1, 1, 2, 2, 130, 64),
                Tensor.uniform(75, -1, 1, 2, 2, 130, 64), null, true, -1.0f), 1e-5, 2e-5);
        same("sdpa BTHD d48 T70 causal", () -> F.scaledDotProductAttentionBTHD(Tensor.uniform(76, -1, 1, 2, 70, 4, 48), Tensor.uniform(77, -1, 1, 2, 70, 4, 48),
                Tensor.uniform(78, -1, 1, 2, 70, 4, 48), null, true, -1.0f), 1e-5, 2e-5);
        same("multi-head attention, packed, d64 T90 causal", () -> F.multiHeadAttention(Tensor.uniform(79, -1, 1, 2, 90, 192), Tensor.uniform(80, -0.1f, 0.1f, 576, 192),
                Tensor.uniform(81, -0.1f, 0.1f, 576), Tensor.uniform(82, -0.1f, 0.1f, 192, 192), null, 3, null, true), 1e-5, 2e-5);
        // the backward: dq, dk, dv concatenated
        same("sdpa d64 T100 S150 + key mask, gradients", () -> {
            Tensor q = Tensor.uniform(83, -1, 1, 2, 3, 100, 64).requiresGrad(true);
            Tensor k = Tensor.uniform(84, -1, 1, 2, 3, 150, 64).requiresGrad(true);
            Tensor v = Tensor.uniform(85, -1, 1, 2, 3, 150, 64).requiresGrad(true);
            F.scaledDotProductAttention(q, k, v, keyMask, false, -1.0f).mul(Tensor.uniform(86, -1, 1, 2, 3, 100, 64)).sum().backward();
            return Tensor.cat(0, q.grad().reshape(q.numel()), k.grad().reshape(k.numel()), v.grad().reshape(v.numel()));
        }, 1e-4, 1e-4);
        same("sdpa d64 T130 causal, gradients", () -> {
            Tensor q = Tensor.uniform(87, -1, 1, 2, 2, 130, 64).requiresGrad(true);
            Tensor k = Tensor.uniform(88, -1, 1, 2, 2, 130, 64).requiresGrad(true);
            Tensor v = Tensor.uniform(89, -1, 1, 2, 2, 130, 64).requiresGrad(true);
            F.scaledDotProductAttention(q, k, v, null, true, -1.0f).mul(Tensor.uniform(90, -1, 1, 2, 2, 130, 64)).sum().backward();
            return Tensor.cat(0, q.grad().reshape(q.numel()), k.grad().reshape(k.numel()), v.grad().reshape(v.numel()));
        }, 1e-4, 1e-4);
        same("multi-head attention, packed, d64 T90 causal, input gradient", () -> {
            Tensor x = Tensor.uniform(91, -1, 1, 2, 90, 192).requiresGrad(true);
            F.multiHeadAttention(x, Tensor.uniform(92, -0.1f, 0.1f, 576, 192), Tensor.uniform(93, -0.1f, 0.1f, 576), Tensor.uniform(94, -0.1f, 0.1f, 192, 192), null, 3, null, true)
                    .mul(Tensor.uniform(95, -1, 1, 2, 90, 192)).sum().backward();
            return x.grad();
        }, 1e-4, 1e-4);
    }

    /** Memory-bound kernels at real sizes: column reductions, LayerNorm and RMSNorm with gradients, broadcasts. */
    @Test
    void memoryBound() {
        same("sum over axis 0 of 2048x768", () -> Tensor.uniform(100, -1, 1, 2048, 768).sum(0, false), 1e-5, 1e-4);
        same("mean over axis 1 of 4x300x50", () -> Tensor.uniform(101, -1, 1, 4, 300, 50).mean(1, false), 1e-5, 2e-5);
        same("max over axis 0 of 500x70", () -> Tensor.uniform(102, -1, 1, 500, 70).max(0, false), 0, 0);
        same("layer norm 1100x768", () -> F.layerNorm(Tensor.uniform(103, -2, 2, 1100, 768), Tensor.uniform(104, 0.5f, 1.5f, 768), Tensor.uniform(105, -1, 1, 768), 1e-5f), 1e-5, 2e-5);
        same("rms norm 300x1000", () -> F.rmsNorm(Tensor.uniform(106, -2, 2, 300, 1000), Tensor.uniform(107, 0.5f, 1.5f, 1000), 1e-6f), 1e-5, 2e-5);
        for (boolean bias : new boolean[] { true, false }) {
            same("layer norm 1100x768 gradients, bias " + bias, () -> {
                Tensor x = Tensor.uniform(108, -2, 2, 1100, 768).requiresGrad(true);
                Tensor w = Tensor.uniform(109, 0.5f, 1.5f, 768).requiresGrad(true);
                Tensor b = bias ? Tensor.uniform(110, -1, 1, 768).requiresGrad(true) : null;
                F.layerNorm(x, w, b, 1e-5f).mul(Tensor.uniform(111, -1, 1, 1100, 768)).sum().backward();
                Tensor g = Tensor.cat(0, x.grad().reshape(x.numel()), w.grad());
                return bias ? Tensor.cat(0, g, b.grad()) : g;
            }, 1e-4, 1e-4);
        }
        same("broadcast add [8,256,768] + [256,768]", () -> Tensor.uniform(112, -1, 1, 8, 256, 768).add(Tensor.uniform(113, -1, 1, 256, 768)));
        same("broadcast mul [16,64,20,20] * [64,1,1]", () -> Tensor.uniform(114, -1, 1, 16, 64, 20, 20).mul(Tensor.uniform(115, -1, 1, 64, 1, 1)));
        same("broadcast sub [768] - [4,30,768]", () -> Tensor.uniform(116, -1, 1, 768).sub(Tensor.uniform(117, -1, 1, 4, 30, 768)));
        same("broadcast add [8,1,768] + [8,256,768] (general path)", () -> Tensor.uniform(118, -1, 1, 8, 1, 768).add(Tensor.uniform(119, -1, 1, 8, 256, 768)));
    }

    /**
     * FP16 autocast inference of Linear: the tensor-core kernel reading the FP32 input (whole
     * tiles), the cast fallback (ragged rows), and the cached weight cast on a second call.
     */
    @Test
    void autocastLinear() {
        for (int rows : new int[] { 512, 300 }) {
            same("autocast FP16 linear " + rows + "x768 -> 1024 + bias + ReLU, twice", () -> Grad.inference(() -> {
                try (Grad.Scope ignored = Autocast.enable(DType.F16)) {
                    // persistent parameters, as a model's: the FP16 cast of the weight is cached and reused by the second trace
                    Tensor w = Tensor.persistent(Tensor.uniform(120, -0.05f, 0.05f, 1024, 768));
                    Tensor b = Tensor.persistent(Tensor.uniform(121, -1, 1, 1024));
                    Tensor y1 = F.linearRelu(Tensor.uniform(122, -1, 1, rows, 768), w, b);
                    float[] first = y1.toFloatArray();
                    Tensor y2 = F.linearRelu(Tensor.uniform(123, -1, 1, rows, 768), w, b);
                    return Tensor.of(first, rows, 1024).add(y2);
                }
            }), 2e-2, 2e-2);
        }
    }

    /** Q8_0 (GGUF blocks): device against the host reference, and the quantisation error against FP32. */
    @Test
    void q8() {
        Supplier<Tensor> x = () -> Tensor.uniform(41, -1, 1, 6, 256);
        Supplier<Tensor> w = () -> Tensor.uniform(42, -0.5f, 0.5f, 300, 256);
        // one quantisation step (max |x| / 127 = 1/127) where rounding a half differs
        same("q8 round trip", () -> F.dequantize(F.quantizeQ8(x.get())), 0, 1.0 / 127 + 1e-6);
        // the products from the same Q8_0 bytes (quantised once on the device: a value on a rounding
        // boundary can land one step apart between devices)
        Tensor qw = Engine.with(Backend.TORNADO, () -> {
            Tensor q = F.quantizeQ8(w.get());
            q.toFloatArray();
            return q;
        });
        same("q8 linear", () -> F.linearQ8(x.get(), qw, Tensor.uniform(43, -1, 1, 300)), 1e-5, 1e-5);
        same("q8 linear batched", () -> F.linearQ8(Tensor.uniform(44, -1, 1, 2, 3, 256), qw, null), 1e-5, 1e-5);
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

    @Test
    void integerCasts() {
        same("FP32 to INT32 to FP32", () -> Tensor.uniform(71, -1000, 1000, 3001).to(DType.I32).to(DType.F32), 0, 0);
        same("INT32 indices from FP32", () -> Tensor.uniform(72, -1, 1, 50, 8).indexSelect(0, Tensor.uniform(73, 0, 49.9f, 20).to(DType.I32)), 0, 0);
    }

    /** LSTM forward and gradients against the reference: fused kernels (H <= 512, B <= 256) and the per-step ones beyond. */
    @Test
    void lstmSequences() {
        // T, B, I, H
        int[][] shapes = { { 200, 8, 3, 48 }, { 37, 5, 7, 512 }, { 12, 260, 4, 16 }, { 9, 3, 6, 513 } };
        for (int[] sh : shapes) {
            int t = sh[0], b = sh[1], in = sh[2], h = sh[3];
            String what = "lstm T" + t + " B" + b + " H" + h;
            java.util.function.Function<Boolean, Supplier<Tensor>> run = padded -> () -> {
                Tensor x = Tensor.uniform(81, -1, 1, t, b, in).requiresGrad(true);
                // PyTorch's initialisation, U(-1/sqrt(H), 1/sqrt(H)): a stable recurrence, so FP32 rounding does not grow
                float k = (float) (1 / Math.sqrt(h));
                Tensor wih = Tensor.uniform(82, -k, k, 4 * h, in).requiresGrad(true);
                Tensor whh = Tensor.uniform(83, -k, k, 4 * h, h).requiresGrad(true);
                Tensor bih = Tensor.uniform(84, -0.1f, 0.1f, 4 * h).requiresGrad(true);
                Tensor bhh = Tensor.uniform(85, -0.1f, 0.1f, 4 * h).requiresGrad(true);
                Tensor h0 = Tensor.uniform(86, -0.5f, 0.5f, b, h).requiresGrad(true);
                Tensor c0 = Tensor.uniform(87, -0.5f, 0.5f, b, h).requiresGrad(true);
                int[] lengths = new int[b];
                for (int i = 0; i < b; i++) {
                    lengths[i] = padded ? Math.max(1, t - (i * 7) % t) : t;
                }
                Tensor y = F.lstm(x, wih, whh, bih, bhh, h0, c0, lengths)[0];
                y.mul(Tensor.uniform(88, -1, 1, t, b, h)).sum().backward();
                // the output and every gradient, side by side
                return Tensor.cat(0, y.reshape(-1), x.grad().reshape(-1), wih.grad().reshape(-1), whh.grad().reshape(-1), bih.grad(), bhh.grad(), h0.grad().reshape(-1),
                        c0.grad().reshape(-1));
            };
            same(what, run.apply(false), 2e-4, 2e-5);
            same(what + " padded", run.apply(true), 2e-4, 2e-5);
        }
    }

    /** 1x1 convolutions on small planes, forward and both gradients (the batched GEMM variants of -Djitnn.conv1x1.*). */
    @Test
    void pointwiseConvolutions() {
        // N, Cin, H, Cout
        int[][] shapes = { { 32, 256, 8, 1024 }, { 32, 1024, 8, 256 }, { 32, 512, 4, 2048 }, { 16, 64, 16, 256 }, { 8, 48, 4, 40 } };
        for (int[] sh : shapes) {
            int n = sh[0], cin = sh[1], hw = sh[2], cout = sh[3];
            String what = "1x1 conv " + n + "x" + cin + "x" + hw + "x" + hw + " -> " + cout;
            same(what, () -> {
                Tensor x = Tensor.uniform(91, -1, 1, n, cin, hw, hw).requiresGrad(true);
                Tensor w = Tensor.uniform(92, -0.1f, 0.1f, cout, cin, 1, 1).requiresGrad(true);
                Tensor y = F.conv2d(x, w, Tensor.uniform(93, -1, 1, cout), 1, 0, 1, 1);
                y.mul(Tensor.uniform(94, -1, 1, n, cout, hw, hw)).sum().backward();
                return Tensor.cat(0, y.reshape(-1), x.grad().reshape(-1), w.grad().reshape(-1));
            }, 1e-4, 1e-4);
        }
    }

    /** Element-wise chains fused into one kernel (inference, and the gradient chains of a backward pass) against the unfused reference. */
    @Test
    void fusedElementwiseChains() {
        same("fused chain", () -> Grad.inference(() -> {
            Tensor x = Tensor.uniform(101, -2, 2, 64, 33);
            Tensor y = Tensor.uniform(102, -2, 2, 64, 33);
            Tensor z = Tensor.uniform(103, 0.5f, 2, 33);
            return x.mul(y).add(1).tanh().mul(z).sigmoid().sub(x.square()).div(z).exp().log1p();
        }));
        same("fused chain, many inputs", () -> Grad.inference(() -> {
            Tensor a = Tensor.uniform(104, -1, 1, 7, 9, 11);
            Tensor b = Tensor.uniform(105, -1, 1, 9, 11);
            Tensor c = Tensor.uniform(106, -1, 1, 11);
            Tensor d = Tensor.uniform(107, -1, 1, 7, 1, 11);
            Tensor e = Tensor.uniform(108, -1, 1, 7, 9, 1);
            return a.add(b).mul(c).relu().add(d).maximum(e).gelu().mul(2).add(Tensor.scalar(0.5f)).abs().sqrt();
        }));
        same("fused chain, intermediates read again", () -> Grad.inference(() -> {
            Tensor x = Tensor.uniform(111, -2, 2, 48, 30);
            Tensor y = Tensor.uniform(112, -2, 2, 30);
            Tensor a = x.mul(y).add(1);
            Tensor b = a.tanh();
            Tensor c = a.sigmoid();
            Tensor d = b.mul(c).add(a);
            return Tensor.cat(0, d.reshape(-1), b.reshape(-1), a.square().reshape(-1));
        }));
        same("fused training chains (forward values the backward reads)", () -> {
            Tensor x = Tensor.uniform(113, -2, 2, 40, 24).requiresGrad(true);
            Tensor w = Tensor.uniform(114, -1, 1, 24).requiresGrad(true);
            Tensor h = x.mul(w).add(0.5f).tanh().mul(x.sigmoid()).gelu();
            h.mul(h).sum().backward();
            return Tensor.cat(0, h.reshape(-1), x.grad().reshape(-1), w.grad());
        }, 1e-5, 1e-5);
        same("fused gradient chains", () -> {
            Tensor x = Tensor.uniform(109, -2, 2, 32, 40).requiresGrad(true);
            Tensor w = Tensor.uniform(110, -1, 1, 40).requiresGrad(true);
            x.mul(w).tanh().mul(x.sigmoid()).add(x.square().mul(0.1f)).gelu().sum().backward();
            return Tensor.cat(0, x.grad().reshape(-1), w.grad());
        }, 1e-5, 1e-5);
    }
}
