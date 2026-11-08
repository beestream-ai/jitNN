package ai.beestream.jitnn.nn.optim;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ops.OptimStep;

/** AdaMax (Kingma and Ba): Adam with an infinity norm, PyTorch's semantics, L2 weight decay. */
public final class AdaMax extends Optimizer {

    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final Tensor[] m;
    private final Tensor[] u;

    public AdaMax(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay) {
        super(params, lr);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.m = new Tensor[this.params.size()];
        this.u = new Tensor[this.params.size()];
        for (int i = 0; i < this.params.size(); i++) {
            int[] dims = this.params.get(i).shape().dims();
            m[i] = Tensor.persistent(Tensor.full(0f, dims));
            u[i] = Tensor.persistent(Tensor.full(0f, dims));
        }
    }

    @Override
    public void step() {
        t++;
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, beta1, beta2, eps, (float) (1 - Math.pow(beta1, t)), weightDecay }, 6);
        List<Tensor> ps = new ArrayList<>();
        List<Tensor> gs = new ArrayList<>();
        List<Tensor> ms = new ArrayList<>();
        List<Tensor> us = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                ps.add(p);
                gs.add(p.grad());
                ms.add(m[i]);
                us.add(u[i]);
            }
        }
        OptimStep.apply(OptimStep.Kind.ADAMAX, ps, gs, List.of(ms, us), hyper, stats);
    }

    @Override
    void rewindStep() {
        t--;
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of("m", m, "u", u);
    }
}
