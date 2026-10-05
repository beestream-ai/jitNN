package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * A stack of LSTM layers over [T, B, I] sequences (time first), PyTorch's {@code nn.LSTM}
 * equations, gate order and initialisation: every weight and bias uniform in
 * [-1/sqrt(H), 1/sqrt(H)). Parameters per layer, in order: W_ih [4H, in], W_hh [4H, H], b_ih,
 * b_hh [4H].
 */
public final class LSTM extends Module {

    /** Output sequence of the last layer and the final states, [layers, B, H] each. */
    public record Output(Tensor y, Tensor h, Tensor c) {
    }

    private final int hidden;
    private final Tensor[][] weights;

    public LSTM(int input, int hidden, int layers, long seed) {
        this.hidden = hidden;
        this.weights = Recurrence.init(this, input, hidden, layers, 4, seed);
    }

    public int hidden() {
        return hidden;
    }

    /** Output sequence of the last layer, from zero initial states. */
    @Override
    public Tensor forward(Tensor x) {
        return forward(x, null, null).y();
    }

    /** From initial states h0, c0 [layers, B, H] (null: zeros); c_n carries no gradient. */
    public Output forward(Tensor x, Tensor h0, Tensor c0) {
        int b = x.dim(1);
        Tensor[] hs = new Tensor[weights.length];
        Tensor[] cs = new Tensor[weights.length];
        Tensor y = x;
        for (int l = 0; l < weights.length; l++) {
            Tensor h = h0 == null ? Tensor.zeros(b, hidden) : h0.narrow(0, l, 1).reshape(b, hidden);
            Tensor c = c0 == null ? Tensor.zeros(b, hidden) : c0.narrow(0, l, 1).reshape(b, hidden);
            Tensor[] w = weights[l];
            Tensor[] r = F.lstm(y, w[0], w[1], w[2], w[3], h, c);
            y = r[0];
            hs[l] = r[1];
            cs[l] = r[2];
        }
        return new Output(y, Tensor.stack(0, hs), Tensor.stack(0, cs));
    }
}
