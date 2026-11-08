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
 * LARS (You et al., 2017) with momentum: a per-parameter learning rate eta |p| / (|g| + weightDecay
 * |p|), applied to the decayed gradient and accumulated in a momentum buffer.
 */
public final class Lars extends Optimizer {

    private final float momentum;
    private final float weightDecay;
    private final float eta;
    private final float eps;
    private final Tensor[] buf;

    /** LARS with the given momentum, weight decay, trust coefficient {@code eta} and {@code eps}. */
    public Lars(List<Tensor> params, float lr, float momentum, float weightDecay, float eta, float eps) {
        super(params, lr);
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.eta = eta;
        this.eps = eps;
        this.buf = new Tensor[this.params.size()];
        for (int i = 0; i < buf.length; i++) {
            buf[i] = Tensor.persistent(Tensor.zeros(this.params.get(i).shape().dims()));
        }
    }

    @Override
    public void step() {
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, momentum, weightDecay, eta, eps }, 5);
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                LayerwiseOptim.lars(p, p.grad(), buf[i], hyper, stats);
            }
        }
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of("buf", buf);
    }
}
