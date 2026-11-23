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

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

import ai.beestream.jitnn.Backend;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.JitnnConfig;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.models.Bert;

/**
 * A BERT encoder in FP32 inference with {@code jitnn.tf32Attention=on}, whose attention reads q, k
 * and v from the packed projection, against the host reference. Run with {@code -Paccel-tests}.
 */
class Tf32AttentionAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    private static float[] encode(Backend backend) {
        return JitnnRuntime.with(backend, () -> {
            Bert bert = new Bert(97, 64, 128, 4, 2, 7);
            bert.train(false);
            int[] ids = new int[2 * 64];
            java.util.Random r = new java.util.Random(3);
            for (int i = 0; i < ids.length; i++) {
                ids[i] = r.nextInt(97);
            }
            return Grad.inference(() -> bert.forward(Tensor.of(ids, 2, 64))).toFloatArray();
        });
    }

    @Test
    void bertMatchesReference() {
        org.junit.jupiter.api.Assumptions.assumeTrue(JitnnRuntime.onCuda(), "the TF32 attention path is the CUDA tensor-core kernel");
        float[] expected = encode(Backend.REFERENCE);
        String before = JitnnConfig.current().get(JitnnConfig.TF32_ATTENTION);
        JitnnConfig.change(JitnnConfig.TF32_ATTENTION, "on");
        try {
            JitnnRuntime.reset();
            float[] got = encode(Backend.TORNADO);
            double max = 0;
            double scale = 1e-6;
            for (int i = 0; i < expected.length; i++) {
                max = Math.max(max, Math.abs(expected[i] - got[i]));
                scale = Math.max(scale, Math.abs(expected[i]));
            }
            System.out.printf("[tf32-attention] bert max error %.2e (scale %.2f)%n", max, scale);
            // FP16-rounded q, k, v through two layers: within 1e-2 of the largest value
            assertTrue(max <= 1e-2 * scale, "max error " + max + " against scale " + scale);
        } finally {
            JitnnConfig.change(JitnnConfig.TF32_ATTENTION, before);
            JitnnRuntime.reset();
        }
    }
}
