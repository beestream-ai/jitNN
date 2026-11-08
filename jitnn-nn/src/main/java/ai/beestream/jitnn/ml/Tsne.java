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
 * Exact t-SNE (van der Maaten and Hinton, 2008) on the device, after scikit-learn's TSNE with
 * method="exact": Gaussian affinities calibrated per point to the perplexity by bisection on the
 * precision, symmetrised; a Student-t kernel in the embedding; gradient descent with momentum
 * (0.5, then 0.8), early exaggeration (12 for 250 iterations), per-parameter gains (+0.2 / x0.8,
 * at least 0.01) and the learning rate max(n / 12 / 4, 50). The embedding starts from PCA, scaled
 * so the first coordinate has standard deviation 1e-4. O(n^2) memory and time per iteration.
 */
public final class Tsne {

    private final int components;
    private final float perplexity;
    private final int iterations;

    /** t-SNE into {@code components} dimensions with the given perplexity and number of iterations. */
    public Tsne(int components, float perplexity, int iterations) {
        this.components = components;
        this.perplexity = perplexity;
        this.iterations = iterations;
    }

    /** t-SNE with scikit-learn's defaults: 2 dimensions, perplexity 30, 1000 iterations. */
    public Tsne() {
        this(2, 30.0f, 1000);
    }

    /** Joint probabilities P [n, n] from squared distances, each row's conditional matched to the perplexity. */
    Tensor affinities(Tensor x) {
        int n = x.dim(0);
        float[] diag = new float[n * n];
        for (int i = 0; i < n; i++) {
            diag[i * n + i] = 1.0f;
        }
        Tensor offDiag = Tensor.of(diag, n, n).neg().add(1.0f);
        Tensor d = Ml.sqDistances(x, x).mul(offDiag);
        float target = (float) Math.log(perplexity);
        Tensor beta = Tensor.ones(n, 1);
        Tensor lo = Tensor.zeros(n, 1);
        Tensor hi = Tensor.zeros(n, 1);
        Tensor hiSet = Tensor.zeros(n, 1);
        Tensor zero = Tensor.scalar(0.0f);
        Tensor p = null;
        for (int it = 0; it < 100; it++) {
            p = d.mul(beta).neg().exp().mul(offDiag);
            Tensor sum = p.sum(1, true).maximum(Tensor.scalar(1e-8f));
            Tensor entropy = sum.log().add(beta.mul(d.mul(p).sum(1, true)).div(sum));
            Tensor diff = entropy.add(-target);
            Tensor done = diff.abs().le(Tensor.scalar(1e-5f));
            Tensor up = diff.gt(zero);
            // entropy too high: precision up (bisect towards hi if known, else double), else down
            Tensor newLo = F.where(up, beta, lo);
            Tensor newHi = F.where(up, hi, beta);
            Tensor newHiSet = F.where(up, hiSet, Tensor.scalar(1.0f).expand(hiSet.shape()));
            Tensor raised = F.where(hiSet.gt(zero), beta.add(hi).mul(0.5f), beta.mul(2.0f));
            Tensor lowered = beta.add(lo).mul(0.5f);
            Tensor nextBeta = F.where(up, raised, lowered);
            beta = F.where(done, beta, nextBeta);
            lo = F.where(done, lo, newLo);
            hi = F.where(done, hi, newHi);
            hiSet = F.where(done, hiSet, newHiSet);
            p = p.div(sum);
            Ml.step();
        }
        Tensor joint = p.add(p.transpose(0, 1));
        return joint.div(joint.sum()).maximum(Tensor.scalar(1.1920929e-7f));
    }

    /** The embedding [n, components] of samples x [n, features], starting from their PCA projection. */
    public Tensor fitTransform(Tensor x) {
        Ml.check(x, "t-SNE");
        int n = x.dim(0);
        Tensor p = affinities(x);
        Tensor y = new Pca(components).fit(x).transform(x);
        float std = (float) Math.sqrt(y.narrow(1, 0, 1).sub(y.narrow(1, 0, 1).mean()).square().mean().item());
        y = y.mul(1e-4f / std);
        float[] eye = new float[n * n];
        for (int i = 0; i < n; i++) {
            eye[i * n + i] = 1.0f;
        }
        Tensor offDiag = Tensor.of(eye, n, n).neg().add(1.0f);
        Tensor update = Tensor.zeros(n, components);
        Tensor gains = Tensor.ones(n, components);
        float lr = Math.max(n / 12.0f / 4.0f, 50.0f);
        Tensor zero = Tensor.scalar(0.0f);
        for (int it = 0; it < iterations; it++) {
            boolean early = it < 250;
            float momentum = early ? 0.5f : 0.8f;
            Tensor pp = early ? p.mul(12.0f) : p;
            Tensor w = Ml.sqDistances(y, y).add(1.0f).reciprocal().mul(offDiag);
            Tensor q = w.div(w.sum()).maximum(Tensor.scalar(1.1920929e-7f));
            Tensor pqw = pp.sub(q).mul(w);
            Tensor grad = pqw.sum(1, true).mul(y).sub(pqw.matmul(y)).mul(4.0f);
            Tensor same = update.mul(grad).lt(zero);
            gains = F.where(same, gains.add(0.2f), gains.mul(0.8f)).maximum(Tensor.scalar(0.01f));
            update = update.mul(momentum).sub(gains.mul(grad).mul(lr));
            y = y.add(update);
            Ml.step();
        }
        return y;
    }
}
