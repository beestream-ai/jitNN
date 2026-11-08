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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.List;
import java.util.function.Supplier;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Autotune;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.models.GPT2;
import ai.beestream.jitnn.nn.models.ResNet;
import ai.beestream.jitnn.nn.optim.Adam;
import ai.beestream.jitnn.nn.optim.Optimizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * With {@link Engine#setDeterministic}, training twice from scratch (plans compiled again, autotune
 * state cleared) gives the same weights bit for bit, with vendor libraries ({@code -Djitnn.native})
 * and autotuning ({@code -Djitnn.autotune=on}) as configured for the run.
 */
class DeterminismAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.setDeterministic(false);
        JitnnRuntime.reset();
    }

    private record Run(Module model, Optimizer opt, Supplier<Tensor> loss) {
    }

    private static float[] train(Supplier<Run> make, int steps) {
        JitnnRuntime.reset();
        Autotune.clear();
        Run r = make.get();
        for (int i = 0; i < steps; i++) {
            Tensor l = r.loss().get();
            l.backward();
            r.opt().step();
            l.item();
            r.opt().zeroGrad();
        }
        List<Tensor> ps = r.model().parameters();
        int n = 0;
        for (Tensor p : ps) {
            n += p.numel();
        }
        float[] all = new float[n];
        int at = 0;
        for (Tensor p : ps) {
            float[] v = p.toFloatArray();
            System.arraycopy(v, 0, all, at, v.length);
            at += v.length;
        }
        return all;
    }

    private static Supplier<Run> cnn() {
        return () -> {
            Module m = ResNet.resnet18(3, 10, true, 5);
            Tensor x = Tensor.uniform(6, -1, 1, 32, 3, 32, 32);
            int[] y = new int[32];
            for (int i = 0; i < y.length; i++) {
                y[i] = i % 10;
            }
            Tensor labels = Tensor.of(y, 32);
            return new Run(m, new Adam(m.parameters(), 1e-3f), () -> F.crossEntropy(m.forward(x), labels));
        };
    }

    private static Supplier<Run> transformer() {
        return () -> {
            GPT2 m = new GPT2(97, 64, 128, 4, 2, 7);
            Tensor ids = ActivationCheckpointRun.tokens(8, 64, 97, 1);
            Tensor next = ActivationCheckpointRun.tokens(8, 64, 97, 2);
            return new Run(m, new Adam(m.parameters(), 1e-3f), () -> ActivationCheckpointRun.lmLoss(m.forward(ids), next));
        };
    }

    @Test
    void sameBitsTwice() {
        for (var c : List.of(cnn(), transformer())) {
            JitnnRuntime.setDeterministic(false);
            float[] a = train(c, 3);
            float[] b = train(c, 3);
            int differ = 0;
            for (int i = 0; i < a.length; i++) {
                differ += Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i]) ? 1 : 0;
            }
            System.out.printf("[determinism] without the switch: %d of %d weights differ between two runs%n", differ, a.length);
            JitnnRuntime.setDeterministic(true);
            assertArrayEquals(train(c, 3), train(c, 3), "deterministic runs differ");
            System.out.println("[determinism] with the switch: identical");
        }
    }
}
