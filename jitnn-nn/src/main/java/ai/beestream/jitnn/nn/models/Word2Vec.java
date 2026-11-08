package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Module;

/**
 * word2vec (Mikolov et al., 2013) with negative sampling: an input embedding (the word vectors)
 * and an output embedding, initialised like the original C code (input uniform in +-0.5 / dim,
 * output zero). Skip-gram predicts a context word from the centre word, CBOW the centre word from
 * the mean of its context. Examples come from {@link ai.beestream.jitnn.nn.text.Corpus}.
 */
public final class Word2Vec extends Module {

    public enum Mode {
        SKIP_GRAM, CBOW
    }

    private final Tensor input;
    private final Tensor output;
    private final Mode mode;
    private final int dim;

    public Word2Vec(int vocab, int dim, Mode mode, long seed) {
        this.mode = mode;
        this.dim = dim;
        this.input = parameter(Tensor.uniform(seed, -0.5f / dim, 0.5f / dim, vocab, dim));
        this.output = parameter(Tensor.zeros(vocab, dim));
    }

    /**
     * The loss of a batch: skip-gram takes centres [B] and contexts [B]; CBOW takes the centre words
     * [B] and their contexts [B * window2] (window2 per sample). negatives holds k words per sample.
     */
    public Tensor loss(int[] centres, int[] contexts, int[] negatives, int k) {
        int b = centres.length;
        Tensor hidden;
        Tensor target;
        if (mode == Mode.SKIP_GRAM) {
            hidden = F.embedding(Tensor.of(centres, b), input);
            target = F.embedding(Tensor.of(contexts, b), output);
        } else {
            int c = contexts.length / b;
            hidden = F.embedding(Tensor.of(contexts, b * c), input).reshape(b, c, dim).mean(1, false);
            target = F.embedding(Tensor.of(centres, b), output);
        }
        Tensor neg = F.embedding(Tensor.of(negatives, b * k), output).reshape(b, k, dim);
        return F.negativeSamplingLoss(hidden, target, neg);
    }

    /** The word vectors of ids x. */
    @Override
    public Tensor forward(Tensor x) {
        return F.embedding(x, input);
    }

    /** All word vectors [vocab, dim] on the host. */
    public float[] vectors() {
        return input.toFloatArray();
    }

    public int dim() {
        return dim;
    }
}
