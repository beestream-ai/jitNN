package ai.beestream.jitnn.nn;

import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

import ai.beestream.jitnn.Buckets;
import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

/**
 * Runs models on inputs padded to {@link Buckets} sizes, so requests of changing batch and
 * sequence length reuse a few compiled plans, and returns results for the real rows and positions
 * only. Inputs are padded and results cut on the host; the values that depend on the real sizes
 * (masks, loss weights) are tensors, not constants, so they never change a plan.
 *
 * <p>Padded rows must not change the real rows: this holds for inference (BatchNorm uses its
 * running statistics) and for training with {@link #crossEntropy} over models without
 * BatchNorm in training mode, whose batch statistics would include the padded rows.
 */
public final class Bucketing {

    /** The additive mask value for padded positions. */
    public static final float MASKED = -1e9f;

    private Bucketing() {
    }

    /**
     * {@code model} over {@code x} [N, ...] with N padded to a {@code batch} size by zero rows;
     * the result's first N rows.
     */
    public static Tensor rows(Buckets batch, Tensor x, UnaryOperator<Tensor> model) {
        int n = x.dim(0);
        return Buckets.cut(model.apply(Buckets.pad(x, 0, batch.size(n), 0)), 0, n);
    }

    /**
     * A causal token model (GPT-2, Llama) over INT32 tokens [B, T]: rows padded to a {@code batch}
     * size, positions padded at the end to a {@code seq} size with {@code padToken}, which the real
     * positions never attend to; the result [B, T, ...] for the real rows and positions.
     */
    public static Tensor causal(Buckets batch, Buckets seq, Tensor idx, int padToken, UnaryOperator<Tensor> model) {
        int b = idx.dim(0), t = idx.dim(1);
        Tensor padded = Buckets.pad(Buckets.pad(idx, 1, seq.size(t), padToken), 0, batch.size(b), padToken);
        return Buckets.cut(Buckets.cut(model.apply(padded), 0, b), 1, t);
    }

    /**
     * An encoder with an additive attention mask (BERT): {@code model} takes tokens [B, T] and a
     * mask [B, 1, 1, T] or [B, 1, T, T] ({@code mask} null for none). Padded positions get
     * {@link #MASKED}, so the real positions do not attend to them; padded rows attend to their
     * padding tokens. The result [B, T, ...] for the real rows and positions.
     */
    public static Tensor masked(Buckets batch, Buckets seq, Tensor idx, Tensor mask, int padToken, BinaryOperator<Tensor> model) {
        int b = idx.dim(0), t = idx.dim(1), bs = batch.size(b), ts = seq.size(t);
        Tensor m = mask != null ? mask : Tensor.of(new float[b * t], b, 1, 1, t);
        int rank = m.rank();
        m = Buckets.pad(m, rank - 1, ts, MASKED);
        if (m.dim(rank - 2) == t && t > 1) {
            m = Buckets.pad(m, rank - 2, ts, 0);
        }
        m = Buckets.pad(m, 0, bs, 0);
        Tensor padded = Buckets.pad(Buckets.pad(idx, 1, ts, padToken), 0, bs, padToken);
        return Buckets.cut(Buckets.cut(model.apply(padded, m), 0, b), 1, t);
    }

    /**
     * Mean cross-entropy of the first N rows of {@code logits} [B, C] (B a padded batch) against
     * INT32 {@code labels} [N]: each real row weighs 1/N and each padded row 0, through a weight
     * tensor, so every N in the bucket runs the same plan. Equal to {@link F#crossEntropy} over
     * the real rows, gradients included.
     */
    public static Tensor crossEntropy(Tensor logits, Tensor labels) {
        int rows = logits.dim(0), n = labels.dim(0);
        float[] w = new float[rows];
        java.util.Arrays.fill(w, 0, n, 1.0f / n);
        Tensor l = Buckets.pad(labels, 0, rows, 0);
        return F.nll(logits.logSoftmax(), l).mul(Tensor.of(w, rows)).sum();
    }
}
