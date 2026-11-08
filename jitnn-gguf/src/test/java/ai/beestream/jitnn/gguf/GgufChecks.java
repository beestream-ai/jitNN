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
package ai.beestream.jitnn.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.ops.Gguf;
import ai.beestream.jitnn.ops.Quant;
import org.junit.jupiter.api.DynamicTest;

/**
 * Checks on {@code fixture.gguf} (written by scripts/gguf/gguf_fixture.py): every weight type, loaded
 * mapped and copied, gives the values llama.cpp's gguf package dequantises it to, and multiplies
 * like them on the given backend.
 */
final class GgufChecks {

    static final String[] TYPES = { "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K", "Q5_0", "Q5_1", "Q2_K", "Q3_K" };

    private GgufChecks() {
    }

    static GgufFile fixture() throws IOException {
        try {
            return GgufFile.open(Path.of(GgufChecks.class.getResource("/gguf/fixture.gguf").toURI()));
        } catch (URISyntaxException e) {
            throw new IOException(e);
        }
    }

    /** One test per weight type and load mode: values, and y = x W^T + b for 1, 5 and 40 rows of x. */
    static List<DynamicTest> weights(Engine.Backend backend) throws IOException {
        GgufFile f = fixture();
        List<DynamicTest> tests = new ArrayList<>();
        for (String type : TYPES) {
            for (boolean copy : new boolean[] { false, true }) {
                tests.add(DynamicTest.dynamicTest(type + (copy ? " copied" : " mapped"), () -> Engine.with(backend, () -> {
                    check(f, type, copy);
                    return null;
                })));
            }
        }
        return tests;
    }

    private static void check(GgufFile f, String type, boolean copy) {
        try {
            Tensor w = f.tensor("w." + type, copy);
            float[] expect = f.tensor("expect." + type, copy).toFloatArray();
            assertEquals(DType.valueOf(type), w.dtype());
            assertEquals(List.of(3, 512), List.of(w.dim(0), w.dim(1)));
            close("values", expect, w.toFloatArray());
            if (w.dtype().isBlockQuantized()) {
                Tensor d = w.dtype() == DType.Q8_0 ? Quant.dequantize(w) : Gguf.dequantize(w);
                close("dequantize", expect, d.toFloatArray());
            }
            Random rnd = new Random(type.hashCode());
            float[] bias = { 0.25f, -0.5f, 1f };
            for (int rows : new int[] { 1, 5, 40 }) {
                float[] x = new float[rows * 512];
                for (int i = 0; i < x.length; i++) {
                    x[i] = rnd.nextFloat() * 2 - 1;
                }
                Tensor xt = Tensor.of(x, rows, 512);
                Tensor b = Tensor.of(bias, 3);
                Tensor y = w.dtype().isBlockQuantized() ? Gguf.linear(xt, w, b) : xt.matmul(w.dtype() == DType.F32 ? w : w.toFloat(), false, true).add(b);
                float[] ref = new float[rows * 3];
                for (int r = 0; r < rows; r++) {
                    for (int n = 0; n < 3; n++) {
                        double s = bias[n];
                        for (int k = 0; k < 512; k++) {
                            s += (double) x[r * 512 + k] * expect[n * 512 + k];
                        }
                        ref[r * 3 + n] = (float) s;
                    }
                }
                close("linear, " + rows + " rows", ref, y.toFloatArray());
            }
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    /** Equal within 2e-4 of the largest magnitude (summation order differs between kernels). */
    static void close(String what, float[] expect, float[] actual) {
        assertEquals(expect.length, actual.length, what + ": length");
        float scale = 1e-6f;
        for (float v : expect) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < expect.length; i++) {
            float diff = Math.abs(expect[i] - actual[i]);
            assertTrue(diff <= 2e-4f * scale, what + "[" + i + "]: expected " + expect[i] + ", got " + actual[i]);
        }
    }
}
