package org.beehive.jitnn.nn;

import org.beehive.jitnn.Tensor;

/** Parameter set-up shared by {@link LSTM} and {@link GRU}. */
final class Recurrence {

    private Recurrence() {
    }

    /** Per layer {W_ih, W_hh, b_ih, b_hh}, uniform in [-1/sqrt(H), 1/sqrt(H)); seeds seed + 4 l + i. */
    static Tensor[][] init(Module owner, int input, int hidden, int layers, int gates, long seed) {
        float bound = (float) (1.0 / Math.sqrt(hidden));
        Tensor[][] w = new Tensor[layers][];
        for (int l = 0; l < layers; l++) {
            int in = l == 0 ? input : hidden;
            long s = seed + 4L * l;
            w[l] = new Tensor[] {
                owner.parameter(Tensor.uniform(s, -bound, bound, gates * hidden, in)),
                owner.parameter(Tensor.uniform(s + 1, -bound, bound, gates * hidden, hidden)),
                owner.parameter(Tensor.uniform(s + 2, -bound, bound, gates * hidden)),
                owner.parameter(Tensor.uniform(s + 3, -bound, bound, gates * hidden)) };
        }
        return w;
    }
}
