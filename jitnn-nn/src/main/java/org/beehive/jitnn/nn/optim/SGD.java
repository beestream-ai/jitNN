package org.beehive.jitnn.nn.optim;

import java.util.List;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ops.OptimStep;

/** Stochastic gradient descent with optional (Nesterov) momentum and L2 weight decay, as in PyTorch. */
public final class SGD extends Optimizer {

    private final float momentum;
    private final float weightDecay;
    private final boolean nesterov;
    private final Tensor[] buffers;

    public SGD(List<Tensor> params, float lr) {
        this(params, lr, 0, 0, false);
    }

    public SGD(List<Tensor> params, float lr, float momentum, float weightDecay, boolean nesterov) {
        super(params, lr);
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.nesterov = nesterov;
        this.buffers = new Tensor[this.params.size()];
        if (momentum != 0) {
            for (int i = 0; i < buffers.length; i++) {
                buffers[i] = Tensor.persistent(Tensor.zeros(this.params.get(i).shape().dims()));
            }
        }
    }

    @Override
    public void step() {
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, momentum, weightDecay, nesterov ? 1 : 0 }, 4);
        List<Tensor> ps = new java.util.ArrayList<>();
        List<Tensor> gs = new java.util.ArrayList<>();
        List<Tensor> bs = new java.util.ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                ps.add(p);
                gs.add(p.grad());
                bs.add(buffers[i]);
            }
        }
        if (momentum != 0) {
            OptimStep.apply(OptimStep.Kind.SGD_MOMENTUM, ps, gs, List.of(bs), hyper, stats);
        } else {
            OptimStep.apply(OptimStep.Kind.SGD, ps, gs, List.of(), hyper, stats);
        }
    }
}
