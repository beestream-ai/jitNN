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

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ops.OptimStep;

/** AdaGrad (Duchi et al.), PyTorch's semantics: learning-rate decay, L2 weight decay, initial accumulator. */
public final class AdaGrad extends Optimizer {

    private final float lrDecay;
    private final float weightDecay;
    private final float eps;
    private final Tensor[] sum;
    private final float init;

    /**
     * AdaGrad with learning-rate decay, L2 weight decay, the accumulator's starting value, and {@code eps}.
     */
    public AdaGrad(List<Tensor> params, float lr, float lrDecay, float weightDecay, float initialAccumulator, float eps) {
        super(params, lr);
        this.init = initialAccumulator;
        this.lrDecay = lrDecay;
        this.weightDecay = weightDecay;
        this.eps = eps;
        this.sum = new Tensor[this.params.size()];
        for (int i = 0; i < this.params.size(); i++) {
            int[] dims = this.params.get(i).shape().dims();
            sum[i] = Tensor.persistent(Tensor.full(init, dims));
        }
    }

    @Override
    public void step() {
        t++;
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr / (1 + (t - 1) * lrDecay), eps, weightDecay }, 3);
        List<Tensor> ps = new ArrayList<>();
        List<Tensor> gs = new ArrayList<>();
        List<Tensor> sums = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                ps.add(p);
                gs.add(p.grad());
                sums.add(sum[i]);
            }
        }
        OptimStep.apply(OptimStep.Kind.ADAGRAD, ps, gs, List.of(sums), hyper, stats);
    }

    @Override
    void rewindStep() {
        t--;
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of("sum", sum);
    }
}
