package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.gradCheck;
import static org.beehive.jitnn.Checks.seq;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.Test;

/** Autograd against finite differences for every differentiable wave 1 operator. */
class GradientTest {

    private static final float[] A23 = seq(6, 1.3f, 0.2f);
    private static final float[] B23 = seq(6, 0.7f, 1.5f);
    private static final float[] B3 = { 0.9f, -1.2f, 1.7f };
    private static final int[] S23 = { 2, 3 };
    private static final int[] S3 = { 3 };

    @Test
    void binaryWithBroadcasting() {
        gradCheck("add", x -> x[0].add(x[1]).square().sum(), new float[][] { A23, B3 }, new int[][] { S23, S3 });
        gradCheck("sub", x -> x[0].sub(x[1]).square().sum(), new float[][] { A23, B3 }, new int[][] { S23, S3 });
        gradCheck("mul", x -> x[0].mul(x[1]).sum(), new float[][] { A23, B3 }, new int[][] { S23, S3 });
        gradCheck("div", x -> x[0].div(x[1]).sum(), new float[][] { A23, B23 }, new int[][] { S23, S23 });
        gradCheck("maximum", x -> x[0].maximum(x[1]).square().sum(), new float[][] { A23, B23 }, new int[][] { S23, S23 });
        gradCheck("col broadcast", x -> x[0].mul(x[1]).square().mean(), new float[][] { A23, { 0.5f, -2 } }, new int[][] { S23, { 2, 1 } });
    }

    @Test
    void unary() {
        float[] pos = { 0.5f, 1.2f, 2.0f, 0.8f, 3.1f, 1.7f };
        gradCheck("exp", x -> x[0].exp().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("log", x -> x[0].log().sum(), new float[][] { pos }, new int[][] { S23 });
        gradCheck("sqrt", x -> x[0].sqrt().sum(), new float[][] { pos }, new int[][] { S23 });
        gradCheck("pow", x -> x[0].pow(2.5f).sum(), new float[][] { pos }, new int[][] { S23 });
        gradCheck("reciprocal", x -> x[0].reciprocal().sum(), new float[][] { pos }, new int[][] { S23 });
        gradCheck("relu", x -> x[0].relu().mul(x[0]).sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("abs", x -> x[0].abs().mul(x[0]).sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("clamp", x -> x[0].clamp(-0.5f, 0.7f).square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("tanh", x -> x[0].tanh().square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("sigmoid", x -> x[0].sigmoid().square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("gelu", x -> x[0].gelu().square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("gelu tanh", x -> x[0].geluTanh().square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("silu", x -> x[0].silu().square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("narrow", x -> x[0].narrow(1, 1, 2).square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("cat", x -> Tensor.cat(1, x[0], x[1]).square().sum(), new float[][] { A23, B23 }, new int[][] { S23, S23 });
        gradCheck("split", x -> x[0].split(1, 1, 2)[1].mul(x[0].split(1, 1, 2)[1]).sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("swiglu", x -> F.swiglu(x[0], x[1]).square().sum(), new float[][] { A23, B23 }, new int[][] { S23, S23 });
        gradCheck("affine+neg", x -> x[0].mul(3).add(1).neg().square().sum(), new float[][] { A23 }, new int[][] { S23 });
    }

    @Test
    void wave4ActivationsAndLosses() {
        float[] prob = { 0.2f, 0.7f, 0.4f, 0.9f, 0.15f, 0.55f };
        float[] target = { 0.3f, 1, 0, 0.6f, 0.2f, 0.9f };
        gradCheck("leaky relu", x -> x[0].leakyRelu(0.1f).square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("softplus", x -> x[0].softplus().square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("mse", x -> F.mseLoss(x[0], x[1]), new float[][] { A23, B23 }, new int[][] { S23, S23 });
        gradCheck("bce", x -> F.binaryCrossEntropy(x[0], x[1]), new float[][] { prob, target }, new int[][] { S23, S23 });
        gradCheck("bce logits", x -> F.binaryCrossEntropyWithLogits(x[0], x[1], F.Reduction.SUM), new float[][] { A23, target }, new int[][] { S23, S23 });
        gradCheck("gaussian kl", x -> F.gaussianKl(x[0], x[1]), new float[][] { A23, B23 }, new int[][] { S23, S23 });
    }

    @Test
    void transposedConvolutionAndResizing() {
        float[] x = seq(18, 0.3f, 0.7f);
        float[] w = seq(54, -0.4f, 0.3f);
        float[] b = { 0.1f, -0.2f, 0.3f };
        gradCheck("conv transpose", t -> F.convTranspose2d(t[0], t[1], t[2], 2, 1, 1, 1, 1).square().sum(), new float[][] { x, w, b },
                new int[][] { { 1, 2, 3, 3 }, { 2, 3, 3, 3 }, { 3 } });
        gradCheck("conv transpose groups", t -> F.convTranspose2d(t[0], t[1], null, 1, 0, 0, 2, 2).square().sum(), new float[][] { x, seq(18, 0.2f, -0.5f) },
                new int[][] { { 1, 2, 3, 3 }, { 2, 1, 3, 3 } });
        gradCheck("upsample nearest", t -> F.upsampleNearest(t[0], 2).square().sum(), new float[][] { x }, new int[][] { { 1, 2, 3, 3 } });
        gradCheck("upsample bilinear", t -> F.upsampleBilinear(t[0], 2, false).square().sum(), new float[][] { x }, new int[][] { { 1, 2, 3, 3 } });
        gradCheck("resize align corners", t -> F.interpolate(t[0], 5, 4, true, true).square().sum(), new float[][] { x }, new int[][] { { 1, 2, 3, 3 } });
    }

    @Test
    void recurrentLayers() {
        float[] x = seq(12, 0.8f, 0.1f);
        float[] h0 = seq(6, 0.5f, -0.1f);
        int[][] lstmShapes = { { 2, 2, 3 }, { 12, 3 }, { 12, 3 }, { 12 }, { 12 }, { 2, 3 }, { 2, 3 } };
        gradCheck("lstm", t -> F.lstm(t[0], t[1], t[2], t[3], t[4], t[5], t[6])[0].square().sum(),
                new float[][] { x, seq(36, 0.4f, 0), seq(36, 0.3f, 0.05f), seq(12, 0.2f, 0), seq(12, 0.2f, 0.1f), h0, seq(6, 0.6f, 0.2f) }, lstmShapes);
        int[][] gruShapes = { { 2, 2, 3 }, { 9, 3 }, { 9, 3 }, { 9 }, { 9 }, { 2, 3 } };
        gradCheck("gru", t -> F.gru(t[0], t[1], t[2], t[3], t[4], t[5])[0].square().sum(),
                new float[][] { x, seq(27, 0.4f, 0), seq(27, 0.3f, 0.05f), seq(9, 0.2f, 0), seq(9, 0.2f, 0.1f), h0 }, gruShapes);
    }

    @Test
    void reshapedParameterGetsItsGradient() {
        Engine.with(Engine.Backend.REFERENCE, () -> {
            Tensor p = Tensor.parameter(Tensor.of(new float[] { 1, -2, 3 }, 3));
            p.reshape(1, 3, 1).mul(Tensor.of(new float[] { 2, 4, 6 }, 1, 3, 1)).sum().backward();
            assertArrayEquals(new float[] { 2, 4, 6 }, p.grad().toFloatArray());
            return null;
        });
    }

    @Test
    void seededRandomMatchesKeyedRandom() {
        Engine.with(Engine.Backend.REFERENCE, () -> {
            long seed = 0x1234_5678_9ABCL;
            assertArrayEquals(Tensor.uniform(seed, -1, 2, 7, 3).toFloatArray(), Tensor.uniform(Tensor.seed(seed), -1, 2, 7, 3).toFloatArray());
            assertArrayEquals(Tensor.normal(seed, 0.5f, 2, 9).toFloatArray(), Tensor.normal(Tensor.seed(seed), 0.5f, 2, 9).toFloatArray());
            return null;
        });
    }

    @Test
    void reductions() {
        gradCheck("sum axis", x -> x[0].sum(0, false).square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("mean keepdim", x -> x[0].mean(1, true).square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("max axis", x -> x[0].max(1, false).square().sum(), new float[][] { A23 }, new int[][] { S23 });
        gradCheck("min axis", x -> x[0].min(0, false).square().sum(), new float[][] { A23 }, new int[][] { S23 });
    }

    @Test
    void matmulAllTransposes() {
        float[] a = seq(12, 1, 0.1f);
        float[] b = seq(20, 0.8f, -0.2f);
        gradCheck("AB", x -> x[0].matmul(x[1]).square().sum(), new float[][] { a, b }, new int[][] { { 3, 4 }, { 4, 5 } });
        gradCheck("AB^T", x -> x[0].matmul(x[1], false, true).square().sum(), new float[][] { a, b }, new int[][] { { 3, 4 }, { 5, 4 } });
        gradCheck("A^TB", x -> x[0].matmul(x[1], true, false).square().sum(), new float[][] { a, b }, new int[][] { { 4, 3 }, { 4, 5 } });
        gradCheck("A^TB^T", x -> x[0].matmul(x[1], true, true).square().sum(), new float[][] { a, b }, new int[][] { { 4, 3 }, { 5, 4 } });
    }

    @Test
    void shapeOps() {
        float[] v = seq(24, 1, 0);
        float[] w = seq(24, 0.5f, 0.3f);
        gradCheck("permute", x -> x[0].permute(2, 0, 1).mul(x[1]).sum(), new float[][] { v, w }, new int[][] { { 2, 3, 4 }, { 4, 2, 3 } });
        gradCheck("reshape", x -> x[0].reshape(6, 4).mul(x[1]).sum(), new float[][] { v, w }, new int[][] { { 2, 3, 4 }, { 6, 4 } });
    }

    @Test
    void softmaxAndLoss() {
        float[] w = seq(6, 0.9f, 0.1f);
        gradCheck("softmax", x -> x[0].softmax().mul(x[1]).sum(), new float[][] { A23, w }, new int[][] { S23, S23 });
        gradCheck("logSoftmax", x -> x[0].logSoftmax().mul(x[1]).sum(), new float[][] { A23, w }, new int[][] { S23, S23 });
        Tensor labels = Tensor.of(new int[] { 2, 0 }, 2);
        gradCheck("crossEntropy", x -> F.crossEntropy(x[0], labels), new float[][] { A23 }, new int[][] { S23 });
    }

    @Test
    void linearLayerChain() {
        float[] x = seq(8, 1, 0);
        float[] w1 = seq(12, 0.6f, 0.05f);
        float[] b1 = { 0.1f, -0.2f, 0.3f };
        float[] w2 = seq(6, 0.8f, -0.1f);
        Tensor labels = Tensor.of(new int[] { 1, 0 }, 2);
        gradCheck("fused linearRelu", t -> F.linearRelu(t[0], t[1], t[2]).square().sum(), new float[][] { x, w1, b1 }, new int[][] { { 2, 4 }, { 3, 4 }, { 3 } });
        gradCheck("mlp", t -> F.crossEntropy(F.linear(F.linear(t[0], t[1], t[2]).relu(), t[3], null), labels),
                new float[][] { x, w1, b1, w2 }, new int[][] { { 2, 4 }, { 3, 4 }, { 3 }, { 2, 3 } });
    }

    @Test
    void convolutionPoolingNorm() {
        float[] x = seq(2 * 4 * 6 * 6, 1, 0);
        float[] w = seq(6 * 4 * 3 * 3, 0.4f, 0.05f);
        float[] wg = seq(6 * 2 * 3 * 3, 0.4f, 0.05f);
        float[] wd = seq(4 * 3 * 3, 0.4f, 0.05f);
        float[] b = { 0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.4f };
        int[] sx = { 2, 4, 6, 6 };
        gradCheck("conv2d", t -> F.conv2d(t[0], t[1], t[2], 1, 1, 1, 1).square().mean(), new float[][] { x, w, b }, new int[][] { sx, { 6, 4, 3, 3 }, { 6 } });
        gradCheck("conv2d strided relu", t -> F.conv2dRelu(t[0], t[1], t[2], 2, 1, 1, 1).square().mean(), new float[][] { x, w, b }, new int[][] { sx, { 6, 4, 3, 3 }, { 6 } });
        gradCheck("conv2d groups dilated", t -> F.conv2d(t[0], t[1], null, 1, 2, 2, 2).square().mean(), new float[][] { x, wg }, new int[][] { sx, { 6, 2, 3, 3 } });
        gradCheck("conv2d depthwise", t -> F.conv2d(t[0], t[1], null, 2, 1, 1, 4).square().mean(), new float[][] { x, wd }, new int[][] { sx, { 4, 1, 3, 3 } });
        // Max pooling needs well-separated values: a finite-difference step must not change the argmax.
        float[] distinct = new float[x.length];
        for (int i = 0; i < distinct.length; i++) {
            distinct[i] = ((i * 37) % distinct.length) * 0.05f - 7;
        }
        gradCheck("max pool", t -> F.maxPool2d(t[0], 3, 2, 1).square().sum(), new float[][] { distinct }, new int[][] { sx });
        gradCheck("avg pool", t -> F.avgPool2d(t[0], 3, 2, 1).square().sum(), new float[][] { x }, new int[][] { sx });
        gradCheck("global avg pool", t -> F.globalAvgPool2d(t[0]).square().sum(), new float[][] { x }, new int[][] { sx });
        float[] gamma = { 1.2f, 0.8f, 1.0f, 0.6f };
        float[] beta = { 0.1f, -0.1f, 0.2f, 0.0f };
        float[] wsum = seq(2 * 4 * 6 * 6, 0.7f, 0.2f);
        Tensor weights = Tensor.of(wsum, sx);
        gradCheck("batch norm", t -> F.batchNorm(t[0], Tensor.persistent(Tensor.zeros(4)), Tensor.persistent(Tensor.ones(4)), t[1], t[2], true, 0.1f, 1e-5f).mul(weights).sum(),
                new float[][] { x, gamma, beta }, new int[][] { sx, { 4 }, { 4 } });
        gradCheck("relu6", t -> t[0].mul(4).relu6().square().sum(), new float[][] { A23 }, new int[][] { S23 });
    }
}
