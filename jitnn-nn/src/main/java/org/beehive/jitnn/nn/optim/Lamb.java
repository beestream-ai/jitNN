package org.beehive.jitnn.nn.optim;

import java.util.List;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ops.LayerwiseOptim;

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
