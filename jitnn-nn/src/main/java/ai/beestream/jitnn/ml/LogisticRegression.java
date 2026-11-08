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

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * L2-regularised logistic regression fitted with L-BFGS, after scikit-learn's LogisticRegression
 * (lbfgs): it minimises C sum_i loss_i + ||W||^2 / 2 with the intercept unpenalised; a sigmoid for
 * two classes, the multinomial (softmax) loss for more.
 */
public final class LogisticRegression {

    private final float c;
    private final int maxIter;
    private int[] classes;
    private Tensor coef;
    private Tensor intercept;

    /**
     * A classifier with inverse regularisation strength {@code c} and at most {@code maxIter} L-BFGS
     * iterations.
     */
    public LogisticRegression(float c, int maxIter) {
        this.c = c;
        this.maxIter = maxIter;
    }

    /** A classifier with C 1 and 500 iterations. */
    public LogisticRegression() {
        this(1.0f, 500);
    }

    /** Fits samples x [n, features] and labels y. */
    public LogisticRegression fit(Tensor x, int[] y) {
        Ml.check(x, "LogisticRegression");
        int n = x.dim(0);
        int d = x.dim(1);
        classes = Ml.classes(y);
        int[] idx = Ml.encode(y, classes);
        int rows = classes.length == 2 ? 1 : classes.length;
        float lambda = 1.0f / (c * n);
        Tensor labels = Tensor.of(idx, n);
        float[] sign = new float[n];
        for (int i = 0; i < n; i++) {
            sign[i] = idx[i] == 1 ? 1.0f : -1.0f;
        }
        Tensor signs = Tensor.of(sign, n);
        float[] p = Lbfgs.minimise(w -> {
            Tensor wd = w.narrow(1, 0, d);
            Tensor b = w.narrow(1, d, 1).reshape(rows);
            Tensor z = x.matmul(wd, false, true).add(b);
            Tensor data = rows == 1 ? z.reshape(n).mul(signs).neg().softplus().mean() : F.crossEntropy(z, labels);
            return data.add(wd.square().sum().mul(0.5f * lambda));
        }, new float[rows * (d + 1)], new int[] { rows, d + 1 }, maxIter, 1e-7, 10);
        Tensor w = Tensor.of(p, rows, d + 1);
        coef = w.narrow(1, 0, d).mul(1.0f);
        intercept = w.narrow(1, d, 1).reshape(rows);
        return this;
    }

    /** Coefficients [1, d] for two classes, [c, d] otherwise. */
    public Tensor coef() {
        return coef;
    }

    /** The intercepts, one per row of the weights. */
    public Tensor intercept() {
        return intercept;
    }

    /** The decision values x W^T + b of each sample. */
    public Tensor decisionFunction(Tensor x) {
        return x.matmul(coef, false, true).add(intercept);
    }

    /** Class probabilities [m, c]. */
    public Tensor predictProba(Tensor x) {
        Tensor z = decisionFunction(x);
        if (classes.length == 2) {
            Tensor p1 = z.sigmoid();
            return Tensor.cat(1, p1.neg().add(1.0f), p1);
        }
        return z.softmax();
    }

    /** The most probable label of each sample. */
    public int[] predict(Tensor x) {
        return Ml.decode(predictProba(x).argmax(1).toIntArray(), classes);
    }
}
