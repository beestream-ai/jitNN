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
