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
package ai.beestream.jitnn.ml;

import ai.beestream.jitnn.Tensor;

/**
 * Linear support vector classification with the squared hinge loss, one-vs-rest, after
 * scikit-learn's LinearSVC (liblinear, intercept_scaling 1): for each class it minimises
 * ||[w, b]||^2 / 2 + C sum_i max(0, 1 - y_i (w . x_i + b))^2, the intercept penalised like a
 * weight, with L-BFGS (the objective is smooth).
 */
public final class LinearSvc {

    private final float c;
    private final int maxIter;
    private int[] classes;
    private Tensor coef;
    private Tensor intercept;

    /** A classifier with penalty {@code c} and at most {@code maxIter} iterations. */
    public LinearSvc(float c, int maxIter) {
        this.c = c;
        this.maxIter = maxIter;
    }

    /** A classifier with scikit-learn's defaults: C 1, 1000 iterations. */
    public LinearSvc() {
        this(1.0f, 1000);
    }

    /** Fits samples x [n, features] and labels y. */
    public LinearSvc fit(Tensor x, int[] y) {
        Ml.check(x, "LinearSvc");
        int n = x.dim(0);
        int d = x.dim(1);
        classes = Ml.classes(y);
        int[] idx = Ml.encode(y, classes);
        int rows = classes.length == 2 ? 1 : classes.length;
        float[] sign = new float[n * rows];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < rows; k++) {
                int positive = rows == 1 ? 1 : k;
                sign[i * rows + k] = idx[i] == positive ? 1.0f : -1.0f;
            }
        }
        Tensor signs = Tensor.of(sign, n, rows);
        float scale = 1.0f / n;
        float[] p = Lbfgs.minimise(w -> {
            Tensor wd = w.narrow(1, 0, d);
            Tensor b = w.narrow(1, d, 1).reshape(rows);
            Tensor margin = x.matmul(wd, false, true).add(b).mul(signs).neg().add(1.0f).relu();
            return margin.square().sum().mul(c * scale).add(w.square().sum().mul(0.5f * scale));
        }, new float[rows * (d + 1)], new int[] { rows, d + 1 }, maxIter, 1e-7, 10);
        Tensor w = Tensor.of(p, rows, d + 1);
        coef = w.narrow(1, 0, d).mul(1.0f);
        intercept = w.narrow(1, d, 1).reshape(rows);
        return this;
    }

    /** The weights: [1, features] for two classes, else [classes, features]. */
    public Tensor coef() {
        return coef;
    }

    /** The intercepts, one per row of {@link #coef()}. */
    public Tensor intercept() {
        return intercept;
    }

    /** The decision values x w^T + b of each sample. */
    public Tensor decisionFunction(Tensor x) {
        return x.matmul(coef, false, true).add(intercept);
    }

    /** The predicted label of each sample. */
    public int[] predict(Tensor x) {
        Tensor z = decisionFunction(x);
        if (classes.length == 2) {
            float[] v = z.toFloatArray();
            int[] out = new int[v.length];
            for (int i = 0; i < v.length; i++) {
                out[i] = classes[v[i] > 0 ? 1 : 0];
            }
            return out;
        }
        return Ml.decode(z.argmax(1).toIntArray(), classes);
    }
}
