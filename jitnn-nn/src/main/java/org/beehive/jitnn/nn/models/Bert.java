package org.beehive.jitnn.nn.models;

import java.nio.file.Path;

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
 * construction order (scripts/pytorch/transformers.py builds the same model). The full form adds
 * BERT's token-type (segment) embeddings and pooler; {@link #fromHuggingFace} builds it.
 */
public final class Bert extends Module {

    private final Tensor decoderBias;
    private final Embedding tokens;
    private final Embedding positions;
    private final LayerNorm embNorm;
    private final Block[] blocks;
    private final Linear headDense;
    private final LayerNorm headNorm;
    private final Embedding types;
    private final Linear pooler;

    public Bert(int vocab, int context, int dim, int heads, int layers, long seed) {
        this(vocab, context, dim, heads, layers, 4 * dim, 0, false, seed);
    }

    /**
     * A BERT with an MLP of {@code hidden} units, {@code typeVocab} token types (0 for none) and,
     * optionally, the pooler (dense and tanh on the first token). The extra parameters are registered
     * after the others, so the short constructor's parameters and seeds are unchanged.
     */
    public Bert(int vocab, int context, int dim, int heads, int layers, int hidden, int typeVocab, boolean pooler, long seed) {
        Seeds seeds = new Seeds(seed);
        this.decoderBias = parameter(Tensor.zeros(vocab));
        this.tokens = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, vocab, dim), -1));
        this.positions = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, context, dim), -1));
        this.embNorm = child(new LayerNorm(dim, 1e-12f, true));
        this.blocks = new Block[layers];
        for (int i = 0; i < layers; i++) {
            blocks[i] = child(new Block(dim, heads, hidden, seeds));
        }
        this.headDense = child(new Linear(dim, dim, seeds.next()));
        this.headNorm = child(new LayerNorm(dim, 1e-12f, true));
        this.types = typeVocab > 0 ? child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, typeVocab, dim), -1)) : null;
        this.pooler = pooler ? child(new Linear(dim, dim, seeds.next())) : null;
    }

    /**
     * BERT from a Hugging Face model directory ({@code bert-base-uncased} and other BERT checkpoints
     * saved as {@code BertModel}, {@code BertForMaskedLM} or {@code BertForPreTraining}): config.json
     * and safetensors weights. The MLM head is loaded when the checkpoint has it; otherwise it keeps
     * its initial values. Task heads (classification and the like) are left to the caller, who can
     * read them with {@link HfCheckpoint}.
     */
    public static Bert fromHuggingFace(Path dir) {
        HfCheckpoint w = HfCheckpoint.open(dir);
        String act = w.configString("hidden_act", "gelu");
        if (!act.equals("gelu")) {
            throw new IllegalArgumentException("hidden_act " + act + " is not supported (BERT uses gelu)");
        }
        if (w.configFloat("layer_norm_eps", 1e-12f) != 1e-12f) {
            throw new IllegalArgumentException("layer_norm_eps other than 1e-12 is not supported");
        }
        if (!w.configString("position_embedding_type", "absolute").equals("absolute")) {
            throw new IllegalArgumentException("only absolute position embeddings are supported");
        }
        int dim = w.configInt("hidden_size");
        Bert m = new Bert(w.configInt("vocab_size"), w.configInt("max_position_embeddings"), dim, w.configInt("num_attention_heads"), w.configInt("num_hidden_layers"),
                w.configInt("intermediate_size"), w.configInt("type_vocab_size"), w.has("pooler.dense.weight"), 0);
        String e = "embeddings.";
        w.load(m.tokens.weight(), e + "word_embeddings.weight");
        w.load(m.positions.weight(), e + "position_embeddings.weight");
        w.load(m.types.weight(), e + "token_type_embeddings.weight");
        w.load(m.embNorm.weight(), e + "LayerNorm.weight");
        w.load(m.embNorm.bias(), e + "LayerNorm.bias");
        for (int i = 0; i < m.blocks.length; i++) {
            Block b = m.blocks[i];
            String l = "encoder.layer." + i + ".";
            w.loadConcatenated(b.qkv.weight(), l + "attention.self.query.weight", l + "attention.self.key.weight", l + "attention.self.value.weight");
            w.loadConcatenated(b.qkv.bias(), l + "attention.self.query.bias", l + "attention.self.key.bias", l + "attention.self.value.bias");
            w.load(b.proj.weight(), l + "attention.output.dense.weight");
            w.load(b.proj.bias(), l + "attention.output.dense.bias");
            w.load(b.ln1.weight(), l + "attention.output.LayerNorm.weight");
            w.load(b.ln1.bias(), l + "attention.output.LayerNorm.bias");
            w.load(b.fc1.weight(), l + "intermediate.dense.weight");
            w.load(b.fc1.bias(), l + "intermediate.dense.bias");
            w.load(b.fc2.weight(), l + "output.dense.weight");
            w.load(b.fc2.bias(), l + "output.dense.bias");
            w.load(b.ln2.weight(), l + "output.LayerNorm.weight");
            w.load(b.ln2.bias(), l + "output.LayerNorm.bias");
        }
        if (m.pooler != null) {
            w.load(m.pooler.weight(), "pooler.dense.weight");
            w.load(m.pooler.bias(), "pooler.dense.bias");
        }
        if (w.has("cls.predictions.transform.dense.weight")) {
            String c = "cls.predictions.";
            w.load(m.headDense.weight(), c + "transform.dense.weight");
            w.load(m.headDense.bias(), c + "transform.dense.bias");
            w.load(m.headNorm.weight(), c + "transform.LayerNorm.weight");
            w.load(m.headNorm.bias(), c + "transform.LayerNorm.bias");
            w.load(m.decoderBias, w.has(c + "bias") ? c + "bias" : c + "decoder.bias");
        }
        return m;
    }

    /** Hidden states [B, T, dim] for INT32 tokens [B, T], without a padding mask. */
    @Override
    public Tensor forward(Tensor idx) {
        return encode(idx, null);
    }

    /** Hidden states [B, T, dim]; {@code mask} is additive, broadcast to [B, heads, T, T] (e.g. [B, 1, 1, T]). */
    public Tensor encode(Tensor idx, Tensor mask) {
        return encode(idx, null, mask);
    }

    /**
     * Hidden states [B, T, dim] with INT32 token types [B, T] (null for all 0; ignored by a model
     * without token-type embeddings) and an additive {@code mask} (null for none).
     */
    public Tensor encode(Tensor idx, Tensor tokenTypes, Tensor mask) {
        int t = idx.dim(1);
        int[] pos = new int[t];
        for (int i = 0; i < t; i++) {
            pos[i] = i;
        }
        Tensor e = tokens.forward(idx).add(positions.forward(Tensor.of(pos, t)));
        if (types != null) {
            e = e.add(types.forward(tokenTypes != null ? tokenTypes : Tensor.of(new int[idx.numel()], idx.shape().dims())));
        }
        Tensor x = embNorm.forward(e);
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

    /** The pooler output [B, dim]: tanh(dense(hidden state of the first token)); needs the pooler. */
    public Tensor pooled(Tensor hidden) {
        if (pooler == null) {
            throw new IllegalStateException("this BERT has no pooler");
        }
        return pooler.forward(hidden.narrow(1, 0, 1).reshape(hidden.dim(0), hidden.dim(2))).tanh();
    }

    static final class Block extends Module {

        private final Linear qkv;
        private final Linear proj;
        private final LayerNorm ln1;
        private final Linear fc1;
        private final Linear fc2;
        private final LayerNorm ln2;
        private final int heads;

        Block(int dim, int heads, int hidden, Seeds seeds) {
            this.heads = heads;
            this.qkv = child(new Linear(dim, 3 * dim, seeds.next()));
            this.proj = child(new Linear(dim, dim, seeds.next()));
            this.ln1 = child(new LayerNorm(dim, 1e-12f, true));
            this.fc1 = child(new Linear(dim, hidden, seeds.next()));
            this.fc2 = child(new Linear(hidden, dim, seeds.next()));
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
