package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.models.MobileNetV2;
import org.junit.jupiter.api.Test;

/**
 * Diagnostic, run on demand ({@code -Djitnn.probe=true -Djitnn.mnist=<dir>}): MobileNetV2 step-0
 * gradients on the host reference backend against PyTorch FP64, per parameter.
 */
class GradProbeTest {

    @Test
    void mobileNetReferenceGradients() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.probe"));
        String dir = System.getProperty("jitnn.mnist", "");
        Engine.setBackend(Engine.Backend.REFERENCE);
        Mnist train = Mnist.load(Path.of(dir), true);
        float[] xb = new float[64 * Mnist.PIXELS];
        int[] yb = new int[64];
        train.batch(0, 64, xb, yb);
        Module model = MobileNetV2.mobilenetV2(1, 10, true, 0, 1);
        F.crossEntropy(model.forward(Tensor.of(xb, 64, 1, 28, 28)), Tensor.of(yb, 64)).backward();
        java.nio.DoubleBuffer truth = ByteBuffer.wrap(Files.readAllBytes(Path.of(dir, "mobilenetv2_pytorch.f32.grad0_f64"))).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        java.nio.FloatBuffer err = ByteBuffer.wrap(Files.readAllBytes(Path.of(dir, "mobilenetv2_pytorch.f32.grad0_f32err"))).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        int i = 0;
        for (Tensor p : model.parameters()) {
            float[] g = p.grad().toFloatArray();
            double d = 0;
            double n = 0;
            for (float v : g) {
                double t = truth.get();
                d += (v - t) * (v - t);
                n += t * t;
            }
            System.out.printf("[probe] %d %s ref-vs-f64 %.2e torchf32 %.2e norm %.2e%n", i, p.shape(), Math.sqrt(d) / Math.max(Math.sqrt(n), 1e-30), err.get(i), Math.sqrt(n));
            i++;
        }
    }
}
