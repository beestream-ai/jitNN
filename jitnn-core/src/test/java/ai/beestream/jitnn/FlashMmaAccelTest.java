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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * FP16 attention on the device (the tensor-core kernels on CUDA) against the FP32 host reference on
 * the same FP16-rounded inputs: head dimensions up to 128, lengths that are not multiples of the
 * tiles, causal and an additive mask. Run with {@code -Paccel-tests}.
 */
class FlashMmaAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    private static float[] attention(Backend backend, int b, int h, int t, int s, int d, boolean causal, boolean masked) {
        return JitnnRuntime.with(backend, () -> Grad.inference(() -> {
            Tensor q = Tensor.uniform(1, -1, 1, b, h, t, d).to(DType.F16);
            Tensor k = Tensor.uniform(2, -1, 1, b, h, s, d).to(DType.F16);
            Tensor v = Tensor.uniform(3, -1, 1, b, h, s, d).to(DType.F16);
            Tensor mask = null;
            if (masked) {
                float[] m = new float[t * s];
                java.util.Random r = new java.util.Random(4);
                for (int i = 0; i < m.length; i++) {
                    m[i] = r.nextInt(5) == 0 ? -1e9f : r.nextFloat() - 0.5f;
                }
                mask = Tensor.of(m, t, s);
            }
            if (backend == Backend.REFERENCE) {
                return F.scaledDotProductAttention(q.to(DType.F32), k.to(DType.F32), v.to(DType.F32), mask, causal, -1);
            }
            try (Grad.Scope ignored = Autocast.enableHalf()) {
                return F.scaledDotProductAttention(q, k, v, mask, causal, -1).to(DType.F32);
            }
        }).toFloatArray());
    }

    /** dq, dk and dv (concatenated) of sum(attention * w), on FP16-rounded inputs. */
    private static float[] gradients(Backend backend, int b, int h, int t, int s, int d, boolean causal, boolean masked) {
        return JitnnRuntime.with(backend, () -> {
            Tensor q = Tensor.uniform(1, -1, 1, b, h, t, d).to(DType.F16).to(DType.F32).requiresGrad(true);
            Tensor k = Tensor.uniform(2, -1, 1, b, h, s, d).to(DType.F16).to(DType.F32).requiresGrad(true);
            Tensor v = Tensor.uniform(3, -1, 1, b, h, s, d).to(DType.F16).to(DType.F32).requiresGrad(true);
            Tensor w = Tensor.uniform(5, -1, 1, b, h, t, d);
            Tensor mask = null;
            if (masked) {
                float[] m = new float[t * s];
                java.util.Random r = new java.util.Random(4);
                for (int i = 0; i < m.length; i++) {
                    m[i] = r.nextInt(5) == 0 ? -1e9f : r.nextFloat() - 0.5f;
                }
                mask = Tensor.of(m, t, s);
            }
            Tensor loss;
            if (backend == Backend.REFERENCE) {
                loss = F.scaledDotProductAttention(q, k, v, mask, causal, -1).mul(w).sum();
            } else {
                try (Grad.Scope ignored = Autocast.enableHalf()) {
                    loss = F.scaledDotProductAttention(q.to(DType.F16), k.to(DType.F16), v.to(DType.F16), mask, causal, -1).to(DType.F32).mul(w).sum();
                }
            }
            loss.backward();
            float[] gq = q.grad().toFloatArray();
            float[] gk = k.grad().toFloatArray();
            float[] gv = v.grad().toFloatArray();
            float[] all = new float[gq.length + gk.length + gv.length];
            System.arraycopy(gq, 0, all, 0, gq.length);
            System.arraycopy(gk, 0, all, gq.length, gk.length);
            System.arraycopy(gv, 0, all, gq.length + gk.length, gv.length);
            return all;
        });
    }

    @Test
    void gradientsMatchReference() {
        int[][] lengths = { { 1, 1 }, { 17, 17 }, { 64, 64 }, { 130, 130 }, { 37, 100 } };
        for (int d : new int[] { 8, 16, 32, 40, 64, 72, 128 }) {
            for (int[] ts : lengths) {
                for (int variant = 0; variant < 3; variant++) {
                    boolean causal = variant == 1;
                    boolean masked = variant == 2;
                    String name = "gradients d " + d + " t " + ts[0] + " s " + ts[1] + (causal ? " causal" : "") + (masked ? " mask" : "");
                    close(gradients(Backend.REFERENCE, 2, 3, ts[0], ts[1], d, causal, masked), gradients(Backend.TORNADO, 2, 3, ts[0], ts[1], d, causal, masked), 2e-2, 2e-2,
                            name);
                }
            }
        }
    }

    @Test
    void matchesReference() {
        int[][] lengths = { { 1, 1 }, { 17, 17 }, { 64, 64 }, { 130, 130 }, { 37, 100 } };
        for (int d : new int[] { 8, 16, 32, 40, 64, 72, 128 }) {
            for (int[] ts : lengths) {
                for (int variant = 0; variant < 3; variant++) {
                    boolean causal = variant == 1;
                    boolean masked = variant == 2;
                    String name = "d " + d + " t " + ts[0] + " s " + ts[1] + (causal ? " causal" : "") + (masked ? " mask" : "");
                    close(attention(Backend.REFERENCE, 2, 3, ts[0], ts[1], d, causal, masked), attention(Backend.TORNADO, 2, 3, ts[0], ts[1], d, causal, masked), 1e-2, 1e-2,
                            name);
                }
            }
        }
    }
}
