/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.Mnist;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.models.MobileNetV2;
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
