package ai.beestream.jitnn.nn.optim;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.StateDict;
import ai.beestream.jitnn.ops.OptimStep;

/** NAdam (Dozat), PyTorch's semantics: momentum decay psi (default 0.004), L2 weight decay. */
public final class NAdam extends Optimizer {

    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final float momentumDecay;
    private double muProduct = 1;
    private final Tensor[] m;
    private final Tensor[] v;

    public NAdam(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay, float momentumDecay) {
        super(params, lr);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.momentumDecay = momentumDecay;
        this.m = new Tensor[this.params.size()];
        this.v = new Tensor[this.params.size()];
        for (int i = 0; i < this.params.size(); i++) {
            int[] dims = this.params.get(i).shape().dims();
            m[i] = Tensor.persistent(Tensor.full(0f, dims));
            v[i] = Tensor.persistent(Tensor.full(0f, dims));
        }
    }

    @Override
    public void step() {
        t++;
        double mu = beta1 * (1 - 0.5 * Math.pow(0.96, t * momentumDecay));
        double muNext = beta1 * (1 - 0.5 * Math.pow(0.96, (t + 1) * momentumDecay));
        muProduct *= mu;
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, beta1, beta2, eps, weightDecay, 0, 0, 0, (float) (1 - Math.pow(beta2, t)), (float) (lr * (1 - mu) / (1 - muProduct)), (float) (lr * muNext / (1 - muProduct * muNext)) }, 11);
        List<Tensor> ps = new ArrayList<>();
        List<Tensor> gs = new ArrayList<>();
        List<Tensor> ms = new ArrayList<>();
        List<Tensor> vs = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                ps.add(p);
                gs.add(p.grad());
                ms.add(m[i]);
                vs.add(v[i]);
            }
        }
        OptimStep.apply(OptimStep.Kind.NADAM, ps, gs, List.of(ms, vs), hyper, stats);
    }

    @Override
    void rewindStep() {
        muProduct /= beta1 * (1 - 0.5 * Math.pow(0.96, t * momentumDecay));
        t--;
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of("m", m, "v", v);
    }

    @Override
    protected void saveScalars(StateDict d) {
        d.put("muProduct", Double.toString(muProduct));
    }

    @Override
    protected void loadScalars(StateDict d) {
        muProduct = Double.parseDouble(d.value("muProduct"));
    }
}
