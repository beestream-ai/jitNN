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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.function.Function;

import ai.beestream.jitnn.engine.Engine;

/** Shared assertions and the finite-difference gradient check. */
public final class Checks {

    private Checks() {
    }

    public static void close(float[] expected, float[] actual, double rtol, double atol, String what) {
        assertEquals(expected.length, actual.length, what + ": length");
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.length; i++) {
            if (Float.isNaN(expected[i]) && Float.isNaN(actual[i]) || Float.isInfinite(expected[i]) && expected[i] == actual[i]) {
                continue;
            }
            double err = Math.abs(expected[i] - actual[i]);
            double bound = atol + rtol * Math.abs(expected[i]);
            if (!(err <= bound)) {
                if (err - bound > worst || at < 0) {
                    worst = err - bound;
                    at = i;
                }
            }
        }
        if (at >= 0) {
            fail(String.format("%s: element %d expected %s got %s (rtol %.1e, atol %.1e)", what, at, expected[at], actual[at], rtol, atol));
        }
    }

    public static float[] seq(int n, float scale, float offset) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = offset + scale * (float) Math.sin(1.7 * i + 0.3);
        }
        return v;
    }

    /**
     * Compares autograd gradients of {@code f} (which must return a scalar) with central finite
     * differences, on the reference backend. FP32 storage limits the accuracy of the numeric
     * derivative, hence the loose default tolerance.
     */
    public static void gradCheck(String what, Function<Tensor[], Tensor> f, float[][] values, int[][] shapes) {
        Engine.with(Engine.Backend.REFERENCE, () -> {
            Tensor[] xs = new Tensor[values.length];
            for (int i = 0; i < xs.length; i++) {
                xs[i] = Tensor.of(values[i], shapes[i]).requiresGrad(true);
            }
            Tensor out = f.apply(xs);
            out.backward();
            for (int i = 0; i < xs.length; i++) {
                float[] analytic = xs[i].grad().toFloatArray();
                float[] numeric = new float[values[i].length];
                for (int j = 0; j < values[i].length; j++) {
                    double eps = 1e-2 * Math.max(1.0, Math.abs(values[i][j]));
                    numeric[j] = (float) ((eval(f, values, shapes, i, j, eps) - eval(f, values, shapes, i, j, -eps)) / (2 * eps));
                }
                close(numeric, analytic, 2e-2, 2e-3, what + " d/dx" + i);
            }
            return null;
        });
    }

    private static double eval(Function<Tensor[], Tensor> f, float[][] values, int[][] shapes, int which, int j, double eps) {
        Tensor[] xs = new Tensor[values.length];
        for (int i = 0; i < xs.length; i++) {
            float[] v = values[i].clone();
            if (i == which) {
                v[j] = (float) (v[j] + eps);
            }
            xs[i] = Tensor.of(v, shapes[i]);
        }
        try (Grad.Scope ignored = Grad.disabled()) {
            return f.apply(xs).item();
        }
    }
}
