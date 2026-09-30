package org.beehive.jitnn.nn.optim;

import java.util.List;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ops.OptimStep;

/** Adam (Kingma and Ba) with L2 weight decay, or AdamW (decoupled decay) via {@link #adamW}. */
public final class Adam extends Optimizer {

    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final boolean decoupled;
    private final Tensor[] m;
    private final Tensor[] v;
    private int t;

    public Adam(List<Tensor> params, float lr) {
        this(params, lr, 0.9f, 0.999f, 1e-8f, 0, false);
    }

    public Adam(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay, boolean decoupled) {
        super(params, lr);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.decoupled = decoupled;
        this.m = new Tensor[this.params.size()];
        this.v = new Tensor[this.params.size()];
        for (int i = 0; i < m.length; i++) {
            int[] dims = this.params.get(i).shape().dims();
            m[i] = Tensor.persistent(Tensor.zeros(dims));
            v[i] = Tensor.persistent(Tensor.zeros(dims));
        }
    }

    public static Adam adamW(List<Tensor> params, float lr, float weightDecay) {
        return new Adam(params, lr, 0.9f, 0.999f, 1e-8f, weightDecay, true);
    }

    @Override
    public void step() {
        t++;
        float bc1 = (float) (1 - Math.pow(beta1, t));
        float bc2 = (float) (1 - Math.pow(beta2, t));
        Tensor hyper = Tensor.of(new float[] { lr, beta1, beta2, eps, bc1, bc2, weightDecay, decoupled ? 1 : 0 }, 8);
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                OptimStep.adam(p, p.grad(), m[i], v[i], hyper);
            }
        }
    }
}
