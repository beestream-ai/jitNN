package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/**
 * A stack of LSTM layers over [T, B, I] sequences (time first), optionally bidirectional, PyTorch's
 * {@code nn.LSTM} equations, gate order, parameter order and initialisation: every weight and
 * bias uniform in [-1/sqrt(H), 1/sqrt(H)). Parameters per layer and direction (forward, then
 * reverse): W_ih [4H, in], W_hh [4H, H], b_ih, b_hh [4H]. Padded batches pass their
 * sequence lengths: outputs past a length are 0 and the final states are each sequence's last.
 */
public final class LSTM extends Module {

    /** Output sequence of the last layer [T, B, directions H] and the final hidden states and cell states, [layers directions, B, H] each. */
    public record Output(Tensor y, Tensor h, Tensor c) {
    }

    private final int hidden;
    private final int directions;
    private final Tensor[][] weights;

    public LSTM(int input, int hidden, int layers, long seed) {
        this(input, hidden, layers, false, seed);
    }

    public LSTM(int input, int hidden, int layers, boolean bidirectional, long seed) {
        this.hidden = hidden;
        this.directions = bidirectional ? 2 : 1;
        this.weights = Recurrence.init(this, input, hidden, layers, directions, 4, seed);
    }

    public int hidden() {
        return hidden;
    }

    /** Output sequence of the last layer, from zero initial states. */
    @Override
    public Tensor forward(Tensor x) {
        return forward(x, null, null, null).y();
    }

    /**
     * From initial states [layers directions, B, H] (null: zeros), for sequences of {@code lengths}
     * (null: all T).
     */
    public Output forward(Tensor x, Tensor h0, Tensor c0, int[] lengths) {
        int b = x.dim(1);
        Tensor[] hs = new Tensor[weights.length];
            Tensor[] cs = new Tensor[weights.length];
        Tensor y = x;
        for (int l = 0; l < weights.length / directions; l++) {
            Tensor[] outs = new Tensor[directions];
            for (int d = 0; d < directions; d++) {
                int k = l * directions + d;
                Tensor h = h0 == null ? Tensor.zeros(b, hidden) : h0.narrow(0, k, 1).reshape(b, hidden);
                Tensor c = c0 == null ? Tensor.zeros(b, hidden) : c0.narrow(0, k, 1).reshape(b, hidden);
                Tensor[] w = weights[k];
                Tensor in = d == 0 ? y : F.reverseSequences(y, lengths);
                Tensor[] r = F.lstm(in, w[0], w[1], w[2], w[3], h, c, lengths);
                outs[d] = d == 0 ? r[0] : F.reverseSequences(r[0], lengths);
                hs[k] = r[1];
                cs[k] = r[2];
            }
            y = directions == 1 ? outs[0] : Tensor.cat(2, outs);
        }
        return new Output(y, Tensor.stack(0, hs), Tensor.stack(0, cs));
    }
}
