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
package ai.beestream.jitnn;

import static ai.beestream.jitnn.Checks.close;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.DynamicTest;

/** Runs every golden operator case on one backend: forward values and input gradients against PyTorch. */
final class GoldenOps {

    private GoldenOps() {
    }

    static List<DynamicTest> tests(Engine.Backend backend) {
        List<DynamicTest> tests = new ArrayList<>();
        for (Golden.Case c : Golden.cases()) {
            if (c.op().equals("optim")) {
                continue;
            }
            tests.add(dynamicTest(c.name(), () -> Engine.with(backend, () -> {
                run(c);
                return null;
            })));
        }
        return tests;
    }

    static void run(Golden.Case c) {
        if (c.op().equals("uniform")) {
            float[] expected = c.get("out").f();
            float[] actual = Tensor.uniform(Long.parseLong(c.attrs()[0]), c.fattr(1), c.fattr(2), expected.length).toFloatArray();
            assertArrayEquals(expected, actual, "Philox uniform is bit-exact with the numpy reproduction");
            return;
        }
        List<Tensor> in = new ArrayList<>();
        for (int i = 0; c.get("in" + i) != null; i++) {
            Tensor t = c.get("in" + i).tensor();
            if (c.get("grad" + i) != null) {
                t.requiresGrad(true);
            }
            in.add(t);
        }
        Tensor[] inputs = in.toArray(new Tensor[0]);
        Golden.EXTRAS.get().clear();
        Tensor y = Golden.apply(c, inputs);
        if (y.dtype() == DType.I32) {
            assertArrayEquals(c.get("out").i(), y.toIntArray(), c.name());
            return;
        }
        double[] tf = Golden.tolerance(c, false);
        Golden.Arr w = c.get("w");
        Tensor loss = y.mul(Tensor.of(w.f(), w.dims())).sum();
        boolean anyGrad = false;
        for (int i = 0; i < inputs.length; i++) {
            anyGrad |= c.get("grad" + i) != null;
        }
        if (anyGrad) {
            loss.backward();
        }
        close(c.get("out").f(), y.toFloatArray(), tf[0], tf[1], c.name() + " forward");
        for (var extra : Golden.EXTRAS.get().entrySet()) {
            close(c.get(extra.getKey()).f(), extra.getValue().toFloatArray(), tf[0], tf[1], c.name() + " " + extra.getKey());
        }
        double[] tg = Golden.tolerance(c, true);
        for (int i = 0; i < inputs.length; i++) {
            if (c.get("grad" + i) != null) {
                close(c.get("grad" + i).f(), inputs[i].grad().toFloatArray(), tg[0], tg[1], c.name() + " grad" + i);
            }
        }
    }
}
