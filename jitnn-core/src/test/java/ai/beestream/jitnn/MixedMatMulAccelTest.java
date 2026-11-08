package ai.beestream.jitnn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Products of an FP32 and an FP16 operand (FP16 training's gradient products): on the tensor cores
 * the FP32 operand is rounded to FP16 as it is staged, so the result equals the product of the
 * FP16-rounded operands with FP32 accumulation, for every transpose and with split K.
 */
class MixedMatMulAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void fp32TimesFp16() {
        int[][] shapes = { { 256, 384, 128 }, { 128, 256, 2048 } };
        for (int[] s : shapes) {
            int m = s[0], n = s[1], k = s[2];
            for (boolean ta : new boolean[] { false, true }) {
                for (boolean tb : new boolean[] { false, true }) {
                    Tensor a = Tensor.uniform(1, -1, 1, ta ? k : m, ta ? m : k);
                    Tensor b = Tensor.uniform(2, -1, 1, tb ? n : k, tb ? k : n);
                    float[] got = a.matmul(b.to(DType.F16), ta, tb).toFloatArray();
                    // the reference: both operands rounded to FP16, the product in FP32
                    float[] want = a.to(DType.F16).to(DType.F32).matmul(b.to(DType.F16).to(DType.F32), ta, tb).toFloatArray();
                    assertEquals(want.length, got.length);
                    double err = 0;
                    double scale = 0;
                    for (int i = 0; i < want.length; i++) {
                        err = Math.max(err, Math.abs(want[i] - got[i]));
                        scale = Math.max(scale, Math.abs(want[i]));
                    }
                    assertTrue(err <= 1e-4 * scale, m + "x" + n + "x" + k + " ta=" + ta + " tb=" + tb + ": max error " + err + " of " + scale);
                }
            }
        }
    }
}
