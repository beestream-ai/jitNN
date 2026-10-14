package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.close;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Supplier;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Half-precision inference ({@link Autocast#enableHalf()}): FP16 activations through linear layers,
 * element-wise operators, LayerNorm and attention on the device, against the FP32 host reference.
 * Run with {@code -Paccel-tests}; with {@code -Djitnn.test.extra=-Djitnn.native=auto} the products
 * go to cuBLAS.
 */
class HalfInferenceAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    /** A post-LayerNorm transformer encoder block, as BERT's, on x [b, t, dim]. */
    private static Tensor block(Tensor x, int heads, int seed) {
        int dim = x.dim(2);
        Tensor wqkv = Tensor.persistent(Tensor.uniform(seed, -0.08f, 0.08f, 3 * dim, dim));
        Tensor bqkv = Tensor.persistent(Tensor.uniform(seed + 1, -0.1f, 0.1f, 3 * dim));
        Tensor wo = Tensor.persistent(Tensor.uniform(seed + 2, -0.08f, 0.08f, dim, dim));
        Tensor bo = Tensor.persistent(Tensor.uniform(seed + 3, -0.1f, 0.1f, dim));
        Tensor w1 = Tensor.persistent(Tensor.uniform(seed + 4, -0.08f, 0.08f, 4 * dim, dim));
        Tensor b1 = Tensor.persistent(Tensor.uniform(seed + 5, -0.1f, 0.1f, 4 * dim));
        Tensor w2 = Tensor.persistent(Tensor.uniform(seed + 6, -0.04f, 0.04f, dim, 4 * dim));
        Tensor b2 = Tensor.persistent(Tensor.uniform(seed + 7, -0.1f, 0.1f, dim));
        Tensor g = Tensor.persistent(Tensor.uniform(seed + 8, 0.8f, 1.2f, dim));
        Tensor beta = Tensor.persistent(Tensor.uniform(seed + 9, -0.1f, 0.1f, dim));
        Tensor a = F.multiHeadAttention(x, wqkv, bqkv, wo, bo, heads, null, false);
        Tensor h = F.layerNorm(x.add(a), g, beta, 1e-5f);
        return F.layerNorm(h.add(F.linear(F.linear(h, w1, b1).gelu(), w2, b2)), g, beta, 1e-5f);
    }

    private static float[] run(Backend backend, boolean half, Supplier<Tensor> model) {
        return Engine.with(backend, () -> Grad.inference(() -> {
            if (!half) {
                return model.get();
            }
            try (Grad.Scope ignored = Autocast.enableHalf()) {
                return model.get();
            }
        }).toFloatArray());
    }

    @Test
    void encoderBlocks() {
        int b = 4, t = 64, dim = 256, heads = 4;
        Supplier<Tensor> model = () -> {
            Tensor x = F.layerNorm(Tensor.uniform(1, -1, 1, b, t, dim), Tensor.ones(dim), Tensor.zeros(dim), 1e-5f);
            for (int layer = 0; layer < 2; layer++) {
                x = block(x, heads, 10 + 20 * layer);
            }
            return x;
        };
        float[] expected = run(Backend.REFERENCE, false, model);
        float[] actual = run(Backend.TORNADO, true, model);
        // FP16 activations: about 3 significant digits after two blocks
        close(expected, actual, 2e-2, 2e-2, "two encoder blocks in FP16");
    }

    @Test
    void activationsStayHalf() {
        Engine.with(Backend.TORNADO, () -> Grad.inference(() -> {
            try (Grad.Scope ignored = Autocast.enableHalf()) {
                Tensor x = Tensor.uniform(2, -1, 1, 64, 128);
                Tensor y = F.linear(x, Tensor.persistent(Tensor.uniform(3, -0.1f, 0.1f, 96, 128)), Tensor.uniform(4, -1, 1, 96)).gelu();
                assertEquals(DType.F16, y.dtype(), "linear + GELU output");
                Tensor z = F.layerNorm(y.add(y), Tensor.ones(96), Tensor.zeros(96), 1e-5f);
                assertEquals(DType.F16, z.dtype(), "add + LayerNorm output");
                Tensor s = z.softmax();
                assertEquals(DType.F32, s.dtype(), "softmax (no FP16 form) reads an FP32 copy");
                return s;
            }
        }).toFloatArray());
    }

    @Test
    void halfMatchesAutocastOnReference() {
        // the reference backend runs the same half path (FP16 storage, FP32 maths)
        Supplier<Tensor> model = () -> F.layerNorm(F.linear(Tensor.uniform(5, -1, 1, 32, 64), Tensor.persistent(Tensor.uniform(6, -0.1f, 0.1f, 48, 64)),
                Tensor.uniform(7, -1, 1, 48)).gelu(), Tensor.ones(48), Tensor.zeros(48), 1e-5f);
        close(run(Backend.REFERENCE, false, model), run(Backend.REFERENCE, true, model), 1e-2, 1e-2, "reference half vs FP32");
    }
}
