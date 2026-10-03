package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.close;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Forward values of the wave 1 operators on the reference backend, against hand-computed results. */
class ReferenceOpsTest {

    @BeforeAll
    static void reference() {
        Engine.setBackend(Engine.Backend.REFERENCE);
    }

    @Test
    void broadcastArithmetic() {
        Tensor a = Tensor.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        Tensor b = Tensor.of(new float[] { 10, 20, 30 }, 3);
        assertArrayEquals(new float[] { 11, 22, 33, 14, 25, 36 }, a.add(b).toFloatArray());
        assertArrayEquals(new float[] { -9, -18, -27, -6, -15, -24 }, a.sub(b).toFloatArray());
        Tensor col = Tensor.of(new float[] { 2, 3 }, 2, 1);
        assertArrayEquals(new float[] { 2, 4, 6, 12, 15, 18 }, a.mul(col).toFloatArray());
        assertArrayEquals(new float[] { 0.5f, 1, 1.5f, 4f / 3, 5f / 3, 2 }, a.div(col).toFloatArray());
        assertArrayEquals(new float[] { 3, 4, 5, 6, 7, 8 }, a.add(Tensor.scalar(2)).toFloatArray());
    }

    @Test
    void unary() {
        Tensor x = Tensor.of(new float[] { -2, -0.5f, 0, 1, 4 }, 5);
        assertArrayEquals(new float[] { 0, 0, 0, 1, 4 }, x.relu().toFloatArray());
        assertArrayEquals(new float[] { 2, 0.5f, 0, 1, 4 }, x.abs().toFloatArray());
        assertArrayEquals(new float[] { -1, -0.5f, 0, 1, 1 }, x.clamp(-1, 1).toFloatArray());
        assertArrayEquals(new float[] { -2.5f, -0.25f, 0.5f, 2, 6.5f }, x.mul(1.5f).add(0.5f).toFloatArray());
        close(new float[] { 1, 2, 4 }, Tensor.of(new float[] { 1, 4, 16 }, 3).sqrt().toFloatArray(), 1e-6, 0, "sqrt");
        close(new float[] { 0, 1, 2 }, Tensor.of(new float[] { 1, (float) Math.E, (float) (Math.E * Math.E) }, 3).log().toFloatArray(), 1e-6, 1e-6, "log");
    }

    @Test
    void reductions() {
        Tensor a = Tensor.of(new float[] { 1, 5, 3, 4, 2, 6 }, 2, 3);
        assertEquals(21, a.sum().item());
        assertEquals(3.5f, a.mean().item());
        assertArrayEquals(new float[] { 5, 7, 9 }, a.sum(0, false).toFloatArray());
        assertArrayEquals(new float[] { 9, 12 }, a.sum(1, false).toFloatArray());
        assertArrayEquals(new float[] { 5, 6 }, a.max(1, false).toFloatArray());
        assertArrayEquals(new float[] { 1, 2, 3 }, a.min(0, false).toFloatArray());
        assertArrayEquals(new int[] { 1, 2 }, a.argmax(1).toIntArray());
        assertEquals(Shape.of(2, 1), a.sum(1, true).shape());
    }

    @Test
    void matmulAndTransposes() {
        Tensor a = Tensor.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        Tensor b = Tensor.of(new float[] { 1, 0, 0, 1, 1, 1 }, 3, 2);
        assertArrayEquals(new float[] { 4, 5, 10, 11 }, a.matmul(b).toFloatArray());
        assertArrayEquals(new float[] { 1, 4, 2, 5, 3, 6 }, a.t().toFloatArray());
        assertArrayEquals(a.matmul(b).toFloatArray(), a.t().matmul(b, true, false).toFloatArray());
        assertArrayEquals(a.matmul(b).toFloatArray(), a.matmul(b.t(), false, true).toFloatArray());
        assertArrayEquals(a.matmul(b).toFloatArray(), a.t().matmul(b.t(), true, true).toFloatArray());
        Tensor p = Tensor.of(Checks.seq(24, 1, 0), 2, 3, 4).permute(2, 0, 1);
        assertEquals(Shape.of(4, 2, 3), p.shape());
        assertEquals(Checks.seq(24, 1, 0)[1 * 12 + 2 * 4 + 3], p.toFloatArray()[3 * 6 + 1 * 3 + 2]);
    }

    @Test
    void softmaxAndCrossEntropy() {
        Tensor logits = Tensor.of(new float[] { 1, 2, 3, 1, 1, 1 }, 2, 3);
        float[] p = logits.softmax().toFloatArray();
        double e = Math.exp(1) + Math.exp(2) + Math.exp(3);
        close(new float[] { (float) (Math.exp(1) / e), (float) (Math.exp(2) / e), (float) (Math.exp(3) / e), 1f / 3, 1f / 3, 1f / 3 }, p, 1e-6, 1e-7, "softmax");
        Tensor labels = Tensor.of(new int[] { 2, 0 }, 2);
        double ce = (-(3 - Math.log(e)) + Math.log(3)) / 2;
        assertEquals(ce, F.crossEntropy(logits, labels).item(), 1e-6);
        // Row 2 ties; argmax takes the first index, which is the label.
        assertEquals(1.0f, F.accuracy(logits, labels));
        assertEquals(0.5f, F.accuracy(logits, Tensor.of(new int[] { 2, 1 }, 2)));
    }

    @Test
    void clipGradientNorm() {
        Tensor p1 = Tensor.parameter(Tensor.of(new float[] { 1, 2 }, 2));
        Tensor p2 = Tensor.parameter(Tensor.of(new float[] { 5 }, 1));
        // gradients (3, 4) and (12): global norm 13
        p1.mul(Tensor.of(new float[] { 3, 4 }, 2)).sum().add(p2.mul(12.0f).sum()).backward();
        Tensor norm = Grad.clipNorm(java.util.List.of(p1, p2), 6.5f);
        assertEquals(13.0f, norm.item(), 1e-5f);
        close(new float[] { 1.5f, 2.0f }, p1.grad().toFloatArray(), 1e-6, 1e-6, "clipped p1");
        close(new float[] { 6.0f }, p2.grad().toFloatArray(), 1e-6, 1e-6, "clipped p2");
        // below the limit the gradients stay (the factor is clamped to 1)
        Grad.clipNorm(java.util.List.of(p1, p2), 100.0f);
        close(new float[] { 1.5f, 2.0f }, p1.grad().toFloatArray(), 1e-6, 1e-6, "unclipped p1");
    }
}
