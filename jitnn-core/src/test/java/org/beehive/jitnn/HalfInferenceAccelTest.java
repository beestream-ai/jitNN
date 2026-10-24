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
        // FP32 on the device too: on Metal with MLX, projection biases move into the attention gather and the norms
        close(expected, run(Backend.TORNADO, false, model), 1e-3, 1e-3, "two encoder blocks in FP32");
        float[] actual = run(Backend.TORNADO, true, model);
        // FP16 activations: about 3 significant digits after two blocks
        close(expected, actual, 2e-2, 2e-2, "two encoder blocks in FP16");
    }

    /** Attention alone: lengths that are not multiples of the 32-wide tensor-core tiles, causal and not. */
    @Test
    void attention() {
        for (boolean causal : new boolean[] { false, true }) {
            Supplier<Tensor> model = () -> {
                Tensor x = Tensor.uniform(30, -1, 1, 3, 200, 192);
                return F.multiHeadAttention(x, Tensor.persistent(Tensor.uniform(31, -0.08f, 0.08f, 576, 192)), Tensor.persistent(Tensor.uniform(32, -0.1f, 0.1f, 576)),
                        Tensor.persistent(Tensor.uniform(33, -0.08f, 0.08f, 192, 192)), Tensor.persistent(Tensor.uniform(34, -0.1f, 0.1f, 192)), 3, null, causal);
            };
            close(run(Backend.REFERENCE, false, model), run(Backend.TORNADO, true, model), 2e-2, 2e-2, "attention t200 d64 causal " + causal);
        }
    }

    /** Linear layers followed by each activation the FP16 epilogues fuse: ReLU, exact GELU, tanh GELU. */
    @Test
    void fusedActivations() {
        Supplier<Tensor> x = () -> Tensor.uniform(40, -1, 1, 96, 128);
        java.util.function.Function<Integer, Supplier<Tensor>> layer = kind -> () -> {
            Tensor y = F.linear(x.get(), Tensor.persistent(Tensor.uniform(41, -0.1f, 0.1f, 160, 128)), Tensor.persistent(Tensor.uniform(42, -1, 1, 160)));
            return switch (kind) {
                case 0 -> y.relu();
                case 1 -> y.gelu();
                default -> y.geluTanh();
            };
        };
        String[] names = { "ReLU", "GELU", "tanh GELU" };
        for (int kind = 0; kind < 3; kind++) {
            close(run(Backend.REFERENCE, false, layer.apply(kind)), run(Backend.TORNADO, true, layer.apply(kind)), 1e-2, 1e-2, "linear + " + names[kind]);
        }
    }

    /** A persistent tensor of host values in [lo, hi): constant to the trace, so convolution + BatchNorm fold. */
    private static Tensor constant(int seed, float lo, float hi, int... dims) {
        java.util.Random r = new java.util.Random(seed);
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = lo + (hi - lo) * r.nextFloat();
        }
        return Tensor.persistent(Tensor.of(v, dims));
    }

    private static Tensor convBn(Tensor x, int cout, int k, int stride, int seed, boolean relu) {
        int cin = x.dim(1);
        float bound = (float) (1.5 / Math.sqrt(cin * k * k));
        Tensor w = constant(seed, -bound, bound, cout, cin, k, k);
        Tensor y = F.conv2d(x, w, null, stride, k / 2, 1, 1);
        Tensor mean = constant(seed + 1, -0.1f, 0.1f, cout);
        Tensor var = constant(seed + 2, 0.5f, 1.5f, cout);
        Tensor g = constant(seed + 3, 0.8f, 1.2f, cout);
        Tensor b = constant(seed + 4, -0.1f, 0.1f, cout);
        return relu ? F.batchNormRelu(y, mean, var, g, b, false, 0.1f, 1e-5f) : F.batchNorm(y, mean, var, g, b, false, 0.1f, 1e-5f);
    }

    /**
     * A ResNet bottleneck: convolutions (FP16 on cuDNN, FP32 kernels otherwise), batch norms, the
     * residual add, from an FP32 image through a strided stem.
     */
    @Test
    void convolutionBlock() {
        Supplier<Tensor> model = () -> {
            Tensor x = convBn(Tensor.uniform(50, -1, 1, 2, 3, 32, 32), 64, 7, 2, 51, true);
            Tensor y = convBn(x, 32, 1, 1, 60, true);
            y = convBn(y, 32, 3, 1, 70, true);
            y = convBn(y, 64, 1, 1, 80, false);
            return x.add(y).relu();
        };
        close(run(Backend.REFERENCE, false, model), run(Backend.TORNADO, true, model), 2e-2, 2e-2, "conv + batch norm block in FP16");
    }

    /**
     * ResNet's stem: the 7x7 convolution, its batch norm and ReLU, 3x3 max pooling, then a bottleneck;
     * with cuDNN or MLX the pooling runs channels-last and applies the stem's bias and ReLU to its maxima.
     * Average pooling the same way.
     */
    @Test
    void stemPooling() {
        for (boolean max : new boolean[] { true, false }) {
            Supplier<Tensor> model = () -> {
                Tensor x = convBn(Tensor.uniform(55, -1, 1, 2, 3, 64, 64), 64, 7, 2, 56, true);
                x = max ? F.maxPool2d(x, 3, 2, 1) : F.avgPool2d(x, 3, 2, 1);
                Tensor y = convBn(x, 32, 1, 1, 57, true);
                y = convBn(y, 32, 3, 1, 58, true);
                y = convBn(y, 64, 1, 1, 59, false);
                return x.add(y).relu();
            };
            close(run(Backend.REFERENCE, false, model), run(Backend.TORNADO, true, model), 2e-2, 2e-2, (max ? "max" : "average") + " pooling after the stem in FP16");
        }
    }

    /**
     * A bottleneck whose 3x3 convolution has the shape jitNN runs as Winograd F(4x4) (128 channels,
     * 4 x 32 x 32), with a projection shortcut: with MLX on Metal it runs channels-last between MLX's 1x1
     * convolutions, their biases applied by the Winograd input transform and the residual add; in FP32
     * and FP16 against the FP32 host reference.
     */
    @Test
    void winogradBottleneck() {
        Supplier<Tensor> model = () -> {
            Tensor x = constant(90, -1, 1, 4, 64, 32, 32);
            Tensor y = convBn(x, 128, 1, 1, 91, true);
            y = convBn(y, 128, 3, 1, 92, true);
            y = convBn(y, 64, 1, 1, 93, false);
            // a projection shortcut, as in ResNet's first block of a stage: both biases go to the add
            Tensor shortcut = convBn(x, 64, 1, 1, 94, false);
            return shortcut.add(y).relu();
        };
        float[] expected = run(Backend.REFERENCE, false, model);
        close(expected, run(Backend.TORNADO, false, model), 1e-3, 1e-3, "Winograd bottleneck in FP32");
        close(expected, run(Backend.TORNADO, true, model), 3e-2, 3e-2, "Winograd bottleneck in FP16");
    }

    /** Grouped convolutions have no FP16 form: they read FP32 copies. */
    @Test
    void groupedConvolution() {
        Supplier<Tensor> model = () -> {
            Tensor x = Tensor.uniform(52, -1, 1, 2, 16, 12, 12);
            Tensor w = Tensor.persistent(Tensor.uniform(53, -0.3f, 0.3f, 16, 4, 3, 3));
            return F.conv2d(F.conv2d(x, Tensor.persistent(Tensor.uniform(54, -0.2f, 0.2f, 16, 16, 3, 3)), null, 1, 1, 1, 1), w, null, 1, 1, 1, 4);
        };
        close(run(Backend.REFERENCE, false, model), run(Backend.TORNADO, true, model), 2e-2, 2e-2, "grouped convolution after an FP16 one");
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
