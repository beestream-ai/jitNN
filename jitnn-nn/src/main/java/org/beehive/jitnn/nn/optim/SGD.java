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
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() == null) {
                continue;
            }
            if (momentum != 0) {
                OptimStep.sgdMomentum(p, p.grad(), buffers[i], hyper, stats);
            } else {
                OptimStep.sgd(p, p.grad(), hyper, stats);
            }
        }
    }
}
