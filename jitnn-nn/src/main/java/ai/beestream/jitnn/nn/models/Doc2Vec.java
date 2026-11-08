package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Module;

/**
 * Paragraph vectors (Le and Mikolov, 2014) with negative sampling. PV-DBOW predicts the words of a
 * document from its vector alone; PV-DM predicts a word from the mean of the document vector and
 * the word vectors of its context.
 */
public final class Doc2Vec extends Module {

    public enum Mode {
        PV_DBOW, PV_DM
    }

    private final Tensor docs;
    private final Tensor words;
    private final Tensor output;
    private final Mode mode;
    private final int dim;

    public Doc2Vec(int vocab, int documents, int dim, Mode mode, long seed) {
        this.mode = mode;
        this.dim = dim;
        this.docs = parameter(Tensor.uniform(seed, -0.5f / dim, 0.5f / dim, documents, dim));
        this.words = mode == Mode.PV_DM ? parameter(Tensor.uniform(seed + 1, -0.5f / dim, 0.5f / dim, vocab, dim)) : null;
        this.output = parameter(Tensor.zeros(vocab, dim));
    }

    /**
     * The loss of a batch: document ids [B], the predicted words [B], for PV-DM the context words [B *
     * window2] (null for PV-DBOW), and k negatives per sample.
     */
    public Tensor loss(int[] documents, int[] targets, int[] contexts, int[] negatives, int k) {
        int b = documents.length;
        Tensor hidden = F.embedding(Tensor.of(documents, b), docs);
        if (mode == Mode.PV_DM) {
            int c = contexts.length / b;
            Tensor ctx = F.embedding(Tensor.of(contexts, b * c), words).reshape(b, c, dim).sum(1, false);
            hidden = hidden.add(ctx).mul(1.0f / (c + 1));
        }
        Tensor target = F.embedding(Tensor.of(targets, b), output);
        Tensor neg = F.embedding(Tensor.of(negatives, b * k), output).reshape(b, k, dim);
        return F.negativeSamplingLoss(hidden, target, neg);
    }

    /** The document vectors of ids x. */
    @Override
    public Tensor forward(Tensor x) {
        return F.embedding(x, docs);
    }

    /** All document vectors [documents, dim] on the host. */
    public float[] documentVectors() {
        return docs.toFloatArray();
    }

    public int dim() {
        return dim;
    }
}
