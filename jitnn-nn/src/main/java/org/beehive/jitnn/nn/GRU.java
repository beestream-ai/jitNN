package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * A stack of GRU layers over [T, B, I] sequences (time first), PyTorch's {@code nn.GRU}
 * equations, gate order (r, z, n) and initialisation: every weight and bias uniform in
 * [-1/sqrt(H), 1/sqrt(H)). Parameters per layer, in order: W_ih [3H, in], W_hh [3H, H], b_ih,
 * b_hh [3H].
 */
public final class GRU extends Module {

    /** Output sequence of the last layer and the final hidden states [layers, B, H]. */
    public record Output(Tensor y, Tensor h) {
    }

    private final int hidden;
    private final Tensor[][] weights;

    public GRU(int input, int hidden, int layers, long seed) {
        this.hidden = hidden;
        this.weights = Recurrence.init(this, input, hidden, layers, 3, seed);
    }

    public int hidden() {
        return hidden;
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, null).y();
    }

    /** From initial states h0 [layers, B, H] (null: zeros). */
    public Output forward(Tensor x, Tensor h0) {
        int b = x.dim(1);
        Tensor[] hs = new Tensor[weights.length];
        Tensor y = x;
        for (int l = 0; l < weights.length; l++) {
            Tensor h = h0 == null ? Tensor.zeros(b, hidden) : h0.narrow(0, l, 1).reshape(b, hidden);
            Tensor[] w = weights[l];
            Tensor[] r = F.gru(y, w[0], w[1], w[2], w[3], h);
            y = r[0];
            hs[l] = r[1];
        }
        return new Output(y, Tensor.stack(0, hs));
    }
}
