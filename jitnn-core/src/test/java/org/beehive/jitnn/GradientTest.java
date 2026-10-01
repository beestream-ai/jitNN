package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.gradCheck;
import static org.beehive.jitnn.Checks.seq;

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
        gradCheck("affine+neg", x -> x[0].mul(3).add(1).neg().square().sum(), new float[][] { A23 }, new int[][] { S23 });
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
}
