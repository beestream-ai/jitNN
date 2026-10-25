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

    private boolean checkpointing;

    /**
     * With {@code on}, training keeps no activations inside the blocks: each block runs again in the
     * backward pass ({@link org.beehive.jitnn.Grad#checkpoint}), for less memory and about a third more compute.
     */
    public Bert checkpointing(boolean on) {
        checkpointing = on;
        return this;
    }

    public Bert(int vocab, int context, int dim, int heads, int layers, long seed) {
        this(vocab, context, dim, heads, layers, 4 * dim, 0, false, seed);
    }

    /**
     * A BERT with an MLP of {@code hidden} units, {@code typeVocab} token types (0 for none) and,
     * optionally, the pooler (dense and tanh on the first token). The extra parameters are registered
     * after the others, so the short constructor's parameters and seeds are unchanged.
     */
    public Bert(int vocab, int context, int dim, int heads, int layers, int hidden, int typeVocab, boolean pooler, long seed) {
        this(vocab, context, dim, heads, layers, hidden, typeVocab, pooler, 1e-12f, seed);
    }

    /** As above with LayerNorm epsilon {@code eps} (RoBERTa uses 1e-5). */
    public Bert(int vocab, int context, int dim, int heads, int layers, int hidden, int typeVocab, boolean pooler, float eps, long seed) {
        Seeds seeds = new Seeds(seed);
        this.decoderBias = parameter(Tensor.zeros(vocab));
        this.tokens = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, vocab, dim), -1));
        this.positions = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, context, dim), -1));
        this.embNorm = child(new LayerNorm(dim, eps, true));
        this.blocks = new Block[layers];
        for (int i = 0; i < layers; i++) {
            blocks[i] = child(new Block(dim, heads, hidden, eps, seeds));
        }
        this.headDense = child(new Linear(dim, dim, seeds.next()));
        this.headNorm = child(new LayerNorm(dim, eps, true));
        this.types = typeVocab > 0 ? child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, typeVocab, dim), -1)) : null;
        this.pooler = pooler ? child(new Linear(dim, dim, seeds.next())) : null;
    }

    /**
     * A BERT-family encoder from a Hugging Face model directory: config.json and safetensors weights,
     * by {@code model_type}:
     * <ul>
     * <li>{@code bert} ({@code bert-base-uncased} and other BERT checkpoints, as {@code BertModel},
     * {@code BertForMaskedLM} or {@code BertForPreTraining});</li>
     * <li>{@code roberta} ({@code roberta-base} and the like): positions counted from the padding
     * index + 1, skipping padding, as RoBERTa does; LayerNorm epsilon 1e-5;</li>
     * <li>{@code distilbert}: no token types, no pooler.</li>
     * </ul>
     * The MLM head is loaded when the checkpoint has it; otherwise it keeps its initial values. Task
     * heads (classification and the like) are left to the caller, who can read them with
     * {@link HfCheckpoint}.
     */
    public static Bert fromHuggingFace(Path dir) {
        HfCheckpoint w = HfCheckpoint.open(dir);
        String type = w.configString("model_type", "bert");
        boolean distil = type.equals("distilbert");
        if (!type.equals("bert") && !type.equals("roberta") && !distil) {
            throw new IllegalArgumentException("model_type " + type + " is not supported (bert, roberta, distilbert are)");
        }
        String act = w.configString(distil ? "activation" : "hidden_act", "gelu");
        if (!act.equals("gelu")) {
            throw new IllegalArgumentException("activation " + act + " is not supported (gelu is)");
        }
        if (!w.configString("position_embedding_type", "absolute").equals("absolute")) {
            throw new IllegalArgumentException("only absolute position embeddings are supported");
        }
        float eps = w.configFloat("layer_norm_eps", 1e-12f);
        int dim = w.configInt("hidden_size", "dim");
        int types = distil ? 0 : w.configInt("type_vocab_size");
        Bert m = new Bert(w.configInt("vocab_size"), w.configInt("max_position_embeddings"), dim, w.configInt("num_attention_heads", "n_heads"),
                w.configInt("num_hidden_layers", "n_layers"), w.configInt("intermediate_size", "hidden_dim"), types, w.has("pooler.dense.weight"), eps, 0);
        if (type.equals("roberta")) {
            m.paddingPositions = w.configInt("pad_token_id");
        }
        String e = "embeddings.";
        w.load(m.tokens.weight(), e + "word_embeddings.weight");
        w.load(m.positions.weight(), e + "position_embeddings.weight");
        if (m.types != null) {
            w.load(m.types.weight(), e + "token_type_embeddings.weight");
        }
        w.load(m.embNorm.weight(), e + "LayerNorm.weight");
        w.load(m.embNorm.bias(), e + "LayerNorm.bias");
        for (int i = 0; i < m.blocks.length; i++) {
            Block b = m.blocks[i];
            if (distil) {
                String l = "transformer.layer." + i + ".";
                w.loadConcatenated(b.qkv.weight(), l + "attention.q_lin.weight", l + "attention.k_lin.weight", l + "attention.v_lin.weight");
                w.loadConcatenated(b.qkv.bias(), l + "attention.q_lin.bias", l + "attention.k_lin.bias", l + "attention.v_lin.bias");
                loadLinear(w, b.proj, l + "attention.out_lin");
                loadNorm(w, b.ln1, l + "sa_layer_norm");
                loadLinear(w, b.fc1, l + "ffn.lin1");
                loadLinear(w, b.fc2, l + "ffn.lin2");
                loadNorm(w, b.ln2, l + "output_layer_norm");
            } else {
                String l = "encoder.layer." + i + ".";
                w.loadConcatenated(b.qkv.weight(), l + "attention.self.query.weight", l + "attention.self.key.weight", l + "attention.self.value.weight");
                w.loadConcatenated(b.qkv.bias(), l + "attention.self.query.bias", l + "attention.self.key.bias", l + "attention.self.value.bias");
                loadLinear(w, b.proj, l + "attention.output.dense");
                loadNorm(w, b.ln1, l + "attention.output.LayerNorm");
                loadLinear(w, b.fc1, l + "intermediate.dense");
                loadLinear(w, b.fc2, l + "output.dense");
                loadNorm(w, b.ln2, l + "output.LayerNorm");
            }
        }
        if (m.pooler != null) {
            loadLinear(w, m.pooler, "pooler.dense");
        }
        // the MLM head: dense, GELU, LayerNorm, then the tied token embedding with its own bias
        String[] head = distil ? new String[] { "vocab_transform", "vocab_layer_norm", "vocab_projector.bias" }
                : type.equals("roberta") ? new String[] { "lm_head.dense", "lm_head.layer_norm", "lm_head.bias" }
                        : new String[] { "cls.predictions.transform.dense", "cls.predictions.transform.LayerNorm", "cls.predictions.bias" };
        if (w.has(head[0] + ".weight")) {
            loadLinear(w, m.headDense, head[0]);
            loadNorm(w, m.headNorm, head[1]);
            String bias = w.has(head[2]) ? head[2] : head[2].replace(".bias", ".decoder.bias");
            w.load(m.decoderBias, bias);
        }
        return m;
    }

    private static void loadLinear(HfCheckpoint w, Linear l, String name) {
        w.load(l.weight(), name + ".weight");
        w.load(l.bias(), name + ".bias");
    }

    private static void loadNorm(HfCheckpoint w, LayerNorm n, String name) {
        w.load(n.weight(), name + ".weight");
        w.load(n.bias(), name + ".bias");
    }

    /** RoBERTa's padding index: positions count from it + 1 over the non-padding tokens; -1 for 0, 1, 2, ... */
    private int paddingPositions = -1;

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
        Tensor positionIds;
        if (paddingPositions >= 0) {
            // RoBERTa: padding_idx + 1, + 2, ... over the real tokens; padding keeps padding_idx
            int[] ids = idx.toIntArray();
            int[] pos = new int[ids.length];
            for (int r = 0; r < ids.length / t; r++) {
                int count = 0;
                for (int i = 0; i < t; i++) {
                    int k = r * t + i;
                    pos[k] = ids[k] == paddingPositions ? paddingPositions : paddingPositions + ++count;
                }
            }
            positionIds = Tensor.of(pos, idx.shape().dims());
        } else {
            int[] pos = new int[t];
            for (int i = 0; i < t; i++) {
                pos[i] = i;
            }
            positionIds = Tensor.of(pos, t);
        }
        Tensor e = tokens.forward(idx).add(positions.forward(positionIds));
        if (types != null) {
            e = e.add(types.forward(tokenTypes != null ? tokenTypes : Tensor.of(new int[idx.numel()], idx.shape().dims())));
        }
        Tensor x = embNorm.forward(e);
        for (Block b : blocks) {
            x = checkpointing ? org.beehive.jitnn.Grad.checkpoint(in -> b.forward(in[0], mask), x) : b.forward(x, mask);
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

    /** True when the model has the pooler (a Hugging Face checkpoint without one, such as roberta-base, gives none). */
    public boolean hasPooler() {
        return pooler != null;
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

        Block(int dim, int heads, int hidden, float eps, Seeds seeds) {
            this.heads = heads;
            this.qkv = child(new Linear(dim, 3 * dim, seeds.next()));
            this.proj = child(new Linear(dim, dim, seeds.next()));
            this.ln1 = child(new LayerNorm(dim, eps, true));
            this.fc1 = child(new Linear(dim, hidden, seeds.next()));
            this.fc2 = child(new Linear(hidden, dim, seeds.next()));
            this.ln2 = child(new LayerNorm(dim, eps, true));
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
