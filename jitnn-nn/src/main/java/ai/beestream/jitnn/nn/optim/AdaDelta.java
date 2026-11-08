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

/** AdaDelta (Zeiler), PyTorch's semantics: decay rho, eps, learning rate (default 1), L2 weight decay. */
public final class AdaDelta extends Optimizer {

    private final float rho;
    private final float eps;
    private final float weightDecay;
    private final Tensor[] square;
    private final Tensor[] delta;

    /** AdaDelta with decay {@code rho}, {@code eps} and L2 weight decay. */
    public AdaDelta(List<Tensor> params, float lr, float rho, float eps, float weightDecay) {
        super(params, lr);
        this.rho = rho;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.square = new Tensor[this.params.size()];
        this.delta = new Tensor[this.params.size()];
        for (int i = 0; i < this.params.size(); i++) {
            int[] dims = this.params.get(i).shape().dims();
            square[i] = Tensor.persistent(Tensor.full(0f, dims));
            delta[i] = Tensor.persistent(Tensor.full(0f, dims));
        }
    }

    @Override
    public void step() {
        t++;
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, rho, eps, weightDecay }, 4);
        List<Tensor> ps = new ArrayList<>();
        List<Tensor> gs = new ArrayList<>();
        List<Tensor> squares = new ArrayList<>();
        List<Tensor> deltas = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                ps.add(p);
                gs.add(p.grad());
                squares.add(square[i]);
                deltas.add(delta[i]);
            }
        }
        OptimStep.apply(OptimStep.Kind.ADADELTA, ps, gs, List.of(squares, deltas), hyper, stats);
    }

    @Override
    void rewindStep() {
        t--;
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of("square", square, "delta", delta);
    }
}
