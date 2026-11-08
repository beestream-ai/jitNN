package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/** Local response normalisation across channels, PyTorch's nn.LocalResponseNorm. */
public final class LocalResponseNorm extends Module {

    private final int size;
    private final float alpha;
    private final float beta;
    private final float k;

    public LocalResponseNorm(int size, float alpha, float beta, float k) {
        this.size = size;
        this.alpha = alpha;
        this.beta = beta;
        this.k = k;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.localResponseNorm(x, size, alpha, beta, k);
    }
}
