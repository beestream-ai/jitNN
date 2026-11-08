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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import ai.beestream.jitnn.Autocast;
import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.F;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.models.Bert;
import ai.beestream.jitnn.nn.models.GPT2;
import ai.beestream.jitnn.nn.models.Llama;
import ai.beestream.jitnn.nn.models.ResNet;
import ai.beestream.jitnn.nn.optim.Adam;
import ai.beestream.jitnn.nn.optim.GradScaler;
import ai.beestream.jitnn.nn.optim.Optimizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Training with FP16 activations ({@link Autocast#enableHalf} and a loss scaler) follows FP32
 * training of the same model from the same initial weights and batches: the losses stay close,
 * the parameters stay FP32 with FP32 gradients, and the loss goes down.
 */
class HalfTrainingAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    private record Case(String name, Supplier<Module> model, java.util.function.Function<Module, Tensor> loss) {
    }

    private static Tensor tokens(int b, int t, int vocab, int salt) {
        return ActivationCheckpointRun.tokens(b, t, vocab, salt);
    }

    private static List<Case> cases() {
        Tensor ids = tokens(4, 32, 97, 1);
        Tensor next = tokens(4, 32, 97, 2);
        Tensor images = Tensor.uniform(5, -1, 1, 8, 3, 32, 32);
        int[] y = new int[8];
        for (int i = 0; i < 8; i++) {
            y[i] = i % 10;
        }
        Tensor labels = Tensor.of(y, 8);
        Tensor labels4 = Tensor.of(new int[] { 0, 1, 2, 3 }, 4);
        return List.of(
                new Case("GPT-2", () -> new GPT2(97, 32, 64, 4, 2, 3), m -> ActivationCheckpointRun.lmLoss(m.forward(ids), next)),
                new Case("BERT", () -> new Bert(97, 32, 64, 4, 2, 128, 2, true, 7), m -> {
                    Bert b = (Bert) m;
                    return F.crossEntropy(b.pooled(b.encode(ids, null, null)).narrow(1, 0, 10), labels4);
                }),
                new Case("Llama GQA", () -> new Llama(97, 128, 4, 2, 2, 256, 10000f, 1e-5f, false, false, 9), m -> ActivationCheckpointRun.lmLoss(m.forward(ids), next)),
                new Case("ResNet-18", () -> ResNet.resnet18(3, 10, true, 5), m -> F.crossEntropy(m.forward(images), labels)));
    }

    private static List<Float> train(Case c, boolean half, int steps) {
        Module m = c.model().get();
        Optimizer opt = new Adam(m.parameters(), 1e-3f);
        GradScaler scaler = half ? new GradScaler() : null;
        List<Float> losses = new ArrayList<>();
        for (int s = 0; s < steps; s++) {
            Tensor loss;
            try (var scope = half ? Autocast.enableHalf() : null) {
                loss = c.loss().apply(m);
            }
            if (half) {
                assertEquals(DType.F32, loss.dtype(), c.name() + " loss");
                scaler.scale(loss).backward();
                for (Tensor p : m.parameters()) {
                    if (p.grad() != null) {
                        assertEquals(DType.F32, p.grad().dtype(), c.name() + " parameter gradient");
                    }
                }
                scaler.step(opt);
                losses.add(loss.item());
                scaler.update();
            } else {
                loss.backward();
                opt.step();
                losses.add(loss.item());
            }
            opt.zeroGrad();
        }
        JitnnRuntime.reset();
        return losses;
    }

    @Test
    void followsFp32Training() {
        for (Case c : cases()) {
            List<Float> full = train(c, false, 12);
            List<Float> half = train(c, true, 12);
            // Differences are relative to the loss, but at least to 5% of the first loss: a model that
            // has nearly memorised its batch (ResNet-18 reaches 1e-3 in a few steps) has a loss of about
            // exp(-margin), whose relative error is the absolute error of the logits, a few percent with
            // FP16 activations (PyTorch's autocast drifts 3-9% from FP32 on the same run)
            double floor = 0.05 * Math.abs(full.get(0));
            double worst = 0;
            for (int s = 0; s < full.size(); s++) {
                worst = Math.max(worst, Math.abs(full.get(s) - half.get(s)) / Math.max(Math.abs(full.get(s)), floor));
            }
            System.out.printf("[half] %s: FP32 losses %s%n[half] %s: FP16 losses %s (largest relative difference %.2e)%n", c.name(), full, c.name(), half, worst);
            assertTrue(worst < 0.02, c.name() + ": FP16 training strays from FP32 by " + worst);
            assertTrue(half.get(half.size() - 1) < half.get(0), c.name() + ": the loss does not go down");
        }
    }
}
