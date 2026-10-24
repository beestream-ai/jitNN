package org.beehive.jitnn.nn.optim;

import java.util.List;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ops.OptimStep;

/**
 * Adam (Kingma and Ba) with L2 weight decay, or AdamW (decoupled decay) via {@link #adamW}; with
 * {@code amsgrad}, AMSGrad's running maximum of the second moment (Reddi et al.).
 */
public final class Adam extends Optimizer {

    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final boolean decoupled;
    private final Tensor[] m;
    private final Tensor[] v;
    private final Tensor[] vmax;

    public Adam(List<Tensor> params, float lr) {
        this(params, lr, 0.9f, 0.999f, 1e-8f, 0, false);
    }

    public Adam(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay, boolean decoupled) {
        this(params, lr, beta1, beta2, eps, weightDecay, decoupled, false);
    }

    public Adam(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay, boolean decoupled, boolean amsgrad) {
        super(params, lr);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.decoupled = decoupled;
        this.m = new Tensor[this.params.size()];
        this.v = new Tensor[this.params.size()];
        this.vmax = amsgrad ? new Tensor[this.params.size()] : null;
        for (int i = 0; i < m.length; i++) {
            int[] dims = this.params.get(i).shape().dims();
            m[i] = Tensor.persistent(Tensor.zeros(dims));
            v[i] = Tensor.persistent(Tensor.zeros(dims));
            if (vmax != null) {
                vmax[i] = Tensor.persistent(Tensor.zeros(dims));
            }
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
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, beta1, beta2, eps, bc1, bc2, weightDecay, decoupled ? 1 : 0 }, 8);
        List<Tensor> ps = new java.util.ArrayList<>();
        List<Tensor> gs = new java.util.ArrayList<>();
        List<Tensor> ms = new java.util.ArrayList<>();
        List<Tensor> vs = new java.util.ArrayList<>();
        List<Tensor> xs = new java.util.ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                ps.add(p);
                gs.add(p.grad());
                ms.add(m[i]);
                vs.add(v[i]);
                if (vmax != null) {
                    xs.add(vmax[i]);
                }
            }
        }
        if (vmax != null) {
            OptimStep.apply(OptimStep.Kind.AMSGRAD, ps, gs, List.of(ms, vs, xs), hyper, stats);
        } else {
            OptimStep.apply(OptimStep.Kind.ADAM, ps, gs, List.of(ms, vs), hyper, stats);
        }
    }

    @Override
    void rewindStep() {
        t--;
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return vmax == null ? java.util.Map.of("m", m, "v", v) : java.util.Map.of("m", m, "v", v, "vmax", vmax);
    }
}
