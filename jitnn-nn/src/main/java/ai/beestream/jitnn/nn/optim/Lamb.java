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
package ai.beestream.jitnn.nn.optim;

import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ops.LayerwiseOptim;

/**
 * LAMB (You et al., 2020): Adam's bias-corrected update plus decoupled weight decay, scaled per
 * parameter by the trust ratio |p| / |update|.
 */
public final class Lamb extends Optimizer {

    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final Tensor[] m;
    private final Tensor[] v;

    /** LAMB with the given betas, {@code eps} and decoupled weight decay. */
    public Lamb(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay) {
        super(params, lr);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.m = new Tensor[this.params.size()];
        this.v = new Tensor[this.params.size()];
        for (int i = 0; i < m.length; i++) {
            int[] dims = this.params.get(i).shape().dims();
            m[i] = Tensor.persistent(Tensor.zeros(dims));
            v[i] = Tensor.persistent(Tensor.zeros(dims));
        }
    }

    @Override
    public void step() {
        t++;
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, beta1, beta2, eps, (float) (1 - Math.pow(beta1, t)), (float) (1 - Math.pow(beta2, t)), weightDecay }, 7);
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                LayerwiseOptim.lamb(p, p.grad(), m[i], v[i], hyper, stats, lr);
            }
        }
    }

    @Override
    void rewindStep() {
        t--;
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of("m", m, "v", v);
    }
}
