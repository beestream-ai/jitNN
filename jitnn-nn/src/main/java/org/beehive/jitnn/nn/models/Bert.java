package org.beehive.jitnn.nn.models;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.Embedding;
import org.beehive.jitnn.nn.LayerNorm;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.Module;

/**
 * A BERT-style encoder for masked language modelling: token and learned position embeddings with
 * LayerNorm, post-LayerNorm blocks of bidirectional attention (with an additive padding mask)
 * and a GELU MLP of 4 x dim, and an MLM head (dense, GELU, LayerNorm, then the token embedding
 * as the decoder with its own bias). LayerNorm eps is 1e-12 as in BERT. Seeds are handed out in
 * construction order (scripts/pytorch/transformers.py builds the same model).
 */
public final class Bert extends Module {

    private final Tensor decoderBias;
    private final Embedding tokens;
    private final Embedding positions;
    private final LayerNorm embNorm;
    private final Block[] blocks;
    private final Linear headDense;
    private final LayerNorm headNorm;

    public Bert(int vocab, int context, int dim, int heads, int layers, long seed) {
        Seeds seeds = new Seeds(seed);
        this.decoderBias = parameter(Tensor.zeros(vocab));
        this.tokens = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, vocab, dim), -1));
        this.positions = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, context, dim), -1));
        this.embNorm = child(new LayerNorm(dim, 1e-12f, true));
        this.blocks = new Block[layers];
        for (int i = 0; i < layers; i++) {
            blocks[i] = child(new Block(dim, heads, seeds));
        }
        this.headDense = child(new Linear(dim, dim, seeds.next()));
        this.headNorm = child(new LayerNorm(dim, 1e-12f, true));
    }

    /** Hidden states [B, T, dim] for INT32 tokens [B, T], without a padding mask. */
    @Override
    public Tensor forward(Tensor idx) {
        return encode(idx, null);
    }

    /** Hidden states [B, T, dim]; {@code mask} is additive, broadcast to [B, heads, T, T] (e.g. [B, 1, 1, T]). */
    public Tensor encode(Tensor idx, Tensor mask) {
        int t = idx.dim(1);
        int[] pos = new int[t];
        for (int i = 0; i < t; i++) {
            pos[i] = i;
        }
        Tensor x = embNorm.forward(tokens.forward(idx).add(positions.forward(Tensor.of(pos, t))));
        for (Block b : blocks) {
            x = b.forward(x, mask);
        }
        return x;
    }

    /**
     * MLM logits [M, vocab] at the INT32 flat positions {@code rows} [M] (indices into B * T): the
     * hidden rows are gathered first, so the head runs on the masked positions only.
     */
    public Tensor mlmLogits(Tensor idx, Tensor mask, Tensor rows) {
        Tensor h = encode(idx, mask);
        Tensor picked = F.embedding(rows, h.reshape(-1, h.dim(2)));
        Tensor z = headNorm.forward(headDense.forward(picked).gelu());
        return F.linear(z, tokens.weight(), decoderBias);
    }

    static final class Block extends Module {

        private final Linear qkv;
        private final Linear proj;
        private final LayerNorm ln1;
        private final Linear fc1;
        private final Linear fc2;
        private final LayerNorm ln2;
        private final int heads;

        Block(int dim, int heads, Seeds seeds) {
            this.heads = heads;
            this.qkv = child(new Linear(dim, 3 * dim, seeds.next()));
            this.proj = child(new Linear(dim, dim, seeds.next()));
            this.ln1 = child(new LayerNorm(dim, 1e-12f, true));
            this.fc1 = child(new Linear(dim, 4 * dim, seeds.next()));
            this.fc2 = child(new Linear(4 * dim, dim, seeds.next()));
            this.ln2 = child(new LayerNorm(dim, 1e-12f, true));
        }

        @Override
        public Tensor forward(Tensor x) {
            return forward(x, null);
        }

        Tensor forward(Tensor x, Tensor mask) {
            Tensor a = F.multiHeadAttention(x, qkv.weight(), qkv.bias(), proj.weight(), proj.bias(), heads, mask, false);
            x = ln1.forward(x.add(a));
            return ln2.forward(x.add(fc2.forward(fc1.forward(x).gelu())));
        }
    }
}
