package ai.beestream.jitnn.nn.optim;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ops.OptimStep;

/** RMSProp (Hinton), PyTorch's semantics: smoothing alpha, optional momentum and centring, L2 weight decay. */
public final class RMSProp extends Optimizer {

    private final float alpha;
    private final float eps;
    private final float weightDecay;
    private final float momentum;
    private final boolean centered;
    private final Tensor[] square;
    private final Tensor[] buffer;
    private final Tensor[] average;

    /**
     * RMSProp with smoothing {@code alpha}, {@code eps}, L2 weight decay, momentum (0 for none) and, with
     * {@code centered}, the variance estimated around the mean gradient.
     */
    public RMSProp(List<Tensor> params, float lr, float alpha, float eps, float weightDecay, float momentum, boolean centered) {
        super(params, lr);
        this.alpha = alpha;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.momentum = momentum;
        this.centered = centered;
        this.square = new Tensor[this.params.size()];
        this.buffer = new Tensor[this.params.size()];
        this.average = new Tensor[this.params.size()];
        for (int i = 0; i < this.params.size(); i++) {
            int[] dims = this.params.get(i).shape().dims();
            square[i] = Tensor.persistent(Tensor.full(0f, dims));
            buffer[i] = Tensor.persistent(Tensor.full(0f, dims));
            average[i] = Tensor.persistent(Tensor.full(0f, dims));
        }
    }

    @Override
    public void step() {
        t++;
        Tensor stats = takeStats();
        Tensor hyper = Tensor.of(new float[] { lr, alpha, eps, weightDecay, momentum, centered ? 1 : 0 }, 6);
        List<Tensor> ps = new ArrayList<>();
        List<Tensor> gs = new ArrayList<>();
        List<Tensor> squares = new ArrayList<>();
        List<Tensor> buffers = new ArrayList<>();
        List<Tensor> averages = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p.grad() != null) {
                ps.add(p);
                gs.add(p.grad());
                squares.add(square[i]);
                buffers.add(buffer[i]);
                averages.add(average[i]);
            }
        }
        OptimStep.apply(OptimStep.Kind.RMSPROP, ps, gs, List.of(squares, buffers, averages), hyper, stats);
    }

    @Override
    void rewindStep() {
        t--;
    }

    @Override
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of("square", square, "buffer", buffer, "average", average);
    }
}
