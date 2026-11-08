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
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.function.Supplier;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Inference fusion (ops.Fusion) against the unfused host reference: convolution + BatchNorm
 * (+ ReLU) folded into one convolution, residual add + ReLU, linear + GELU; the folded weights
 * follow BatchNorm statistics that training changes, also when the plan is remembered; a
 * fused-away intermediate cannot be read.
 */
class FusionAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    private static void close(float[] expected, float[] actual, double rtol, double atol, String what) {
        assertEquals(expected.length, actual.length, what);
        for (int i = 0; i < expected.length; i++) {
            double tol = atol + rtol * Math.abs(expected[i]);
            if (Math.abs(expected[i] - actual[i]) > tol) {
                throw new AssertionError(what + ": element " + i + " is " + actual[i] + ", expected " + expected[i]);
            }
        }
    }

    private static float[] on(Backend b, Supplier<Tensor> f) {
        return JitnnRuntime.with(b, () -> f.get().toFloatArray());
    }

    /** A residual block with a projection shortcut, then global pooling and a linear + GELU head. */
    private static final class Net extends Module {
        final Module block = child(new Add(
                new Sequential(new Conv2d(4, 8, 3, 1, 1, 1, 1, false, 1), new BatchNorm2d(8), new ReLU(), new Conv2d(8, 8, 3, 1, 1, 1, 1, true, 2), new BatchNorm2d(8)),
                new Sequential(new Conv2d(4, 8, 1, 1, 0, 1, 1, false, 3), new BatchNorm2d(8)), true));
        final Linear head = child(new Linear(8, 6, 4));

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = block.forward(x);
            return head.forward(h.reshape(h.dim(0), 8, -1).mean(2, false)).gelu();
        }
    }

    @Test
    void fusedInferenceMatchesTheReference() {
        Net net = new Net();
        Tensor x = Tensor.uniform(5, -1, 1, 2, 4, 9, 7);
        float[] xs = on(Backend.REFERENCE, () -> x);
        // give the BatchNorms non-trivial running statistics with a few training steps
        for (int s = 0; s < 3; s++) {
            Tensor xb = Tensor.uniform(10 + s, -2, 2, 2, 4, 9, 7);
            JitnnRuntime.with(Backend.TORNADO, () -> net.forward(xb).sum().item());
        }
        net.train(false);
        Supplier<Tensor> infer = () -> Grad.inference(() -> net.forward(Tensor.of(xs, 2, 4, 9, 7)));
        close(on(Backend.REFERENCE, infer), on(Backend.TORNADO, infer), 1e-5, 1e-5, "fused inference");
        close(on(Backend.REFERENCE, infer), on(Backend.TORNADO, infer), 1e-5, 1e-5, "fused inference, replayed");
        // the trace is remembered once its plan is settled (the folded weights cached): later calls
        // replay the plan without rewriting the trace
        long remembered = JitnnRuntime.planStats()[3];
        for (int i = 0; i < 4 && JitnnRuntime.planStats()[3] == remembered; i++) {
            close(on(Backend.REFERENCE, infer), on(Backend.TORNADO, infer), 1e-5, 1e-5, "fused inference, settling");
        }
        remembered = JitnnRuntime.planStats()[3];
        close(on(Backend.REFERENCE, infer), on(Backend.TORNADO, infer), 1e-5, 1e-5, "fused inference, remembered");
        assertEquals(remembered + 1, JitnnRuntime.planStats()[3], "a remembered replay");
        // training changes the statistics: the folded weights must follow
        net.train(true);
        JitnnRuntime.with(Backend.TORNADO, () -> net.forward(Tensor.uniform(20, -3, 3, 2, 4, 9, 7)).sum().item());
        net.train(false);
        close(on(Backend.REFERENCE, infer), on(Backend.TORNADO, infer), 1e-5, 1e-5, "fused inference after the statistics changed");
    }

    @Test
    void fusedIntermediatesCannotBeRead() {
        Conv2d conv = new Conv2d(3, 5, 3, 1, 1, 1, 1, false, 7);
        BatchNorm2d bn = new BatchNorm2d(5);
        bn.train(false);
        Tensor[] mid = new Tensor[1];
        JitnnRuntime.with(Backend.TORNADO, () -> Grad.inference(() -> {
            mid[0] = conv.forward(Tensor.uniform(8, -1, 1, 1, 3, 6, 6));
            return bn.forward(mid[0]);
        }).toFloatArray());
        assertThrows(IllegalStateException.class, () -> mid[0].toFloatArray());
    }
}
