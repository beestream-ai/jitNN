package org.beehive.jitnn.nn.models;

import java.nio.file.Path;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.Embedding;
import org.beehive.jitnn.nn.LayerNorm;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.Module;

/**
 * A GPT-2 style decoder: token and learned position embeddings, pre-LayerNorm blocks of causal
 * multi-head attention and a GELU (tanh form) MLP of 4 x dim, a final LayerNorm, and logits from
 * the token embedding (tied weights). Embeddings start uniform with standard deviation 0.02 and
 * linear layers as {@link Linear}, with seeds handed out in construction order
 * (scripts/pytorch/transformers.py builds the same model).
 */
public final class GPT2 extends Module {

    private final Embedding tokens;
    private final Embedding positions;
    private final Block[] blocks;
    private final LayerNorm norm;

    private boolean checkpointing;

    /**
     * With {@code on}, training keeps no activations inside the blocks: each block runs again in the
     * backward pass ({@link org.beehive.jitnn.Grad#checkpoint}), for less memory and about a third more compute.
     */
    public GPT2 checkpointing(boolean on) {
        checkpointing = on;
        return this;
    }

    public GPT2(int vocab, int context, int dim, int heads, int layers, long seed) {
        Seeds seeds = new Seeds(seed);
        this.tokens = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, vocab, dim), -1));
        this.positions = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, context, dim), -1));
        this.blocks = new Block[layers];
        for (int i = 0; i < layers; i++) {
            blocks[i] = child(new Block(dim, heads, seeds));
        }
        this.norm = child(new LayerNorm(dim));
    }

    /**
     * GPT-2 from a Hugging Face model directory ({@code gpt2}, {@code gpt2-medium}, ... or a fine-tuned
     * copy): {@code config.json} and safetensors weights, as {@code GPT2Model} or {@code GPT2LMHeadModel}
     * save them. The language-model head is the tied token embedding, as in GPT-2.
     */
    public static GPT2 fromHuggingFace(Path dir) {
        HfCheckpoint w = HfCheckpoint.open(dir);
        int dim = w.configInt("n_embd");
        if (w.config().get("n_inner") instanceof Number inner && inner.intValue() != 4 * dim) {
            throw new IllegalArgumentException("n_inner " + inner + " is not 4 x n_embd");
        }
        String act = w.configString("activation_function", "gelu_new");
        if (!act.equals("gelu_new") && !act.equals("gelu_pytorch_tanh")) {
            throw new IllegalArgumentException("activation " + act + " is not supported (GPT-2 uses gelu_new)");
        }
        if (w.configFloat("layer_norm_epsilon", 1e-5f) != 1e-5f) {
            throw new IllegalArgumentException("layer_norm_epsilon other than 1e-5 is not supported");
        }
        GPT2 m = new GPT2(w.configInt("vocab_size"), w.configInt("n_positions", "n_ctx"), dim, w.configInt("n_head"), w.configInt("n_layer"), 0);
        w.load(m.tokens.weight(), "wte.weight");
        w.load(m.positions.weight(), "wpe.weight");
        for (int i = 0; i < m.blocks.length; i++) {
            Block b = m.blocks[i];
            String h = "h." + i + ".";
            w.load(b.ln1.weight(), h + "ln_1.weight");
            w.load(b.ln1.bias(), h + "ln_1.bias");
            w.loadTransposed(b.qkv.weight(), h + "attn.c_attn.weight");
            w.load(b.qkv.bias(), h + "attn.c_attn.bias");
            w.loadTransposed(b.proj.weight(), h + "attn.c_proj.weight");
            w.load(b.proj.bias(), h + "attn.c_proj.bias");
            w.load(b.ln2.weight(), h + "ln_2.weight");
            w.load(b.ln2.bias(), h + "ln_2.bias");
            w.loadTransposed(b.fc.weight(), h + "mlp.c_fc.weight");
            w.load(b.fc.bias(), h + "mlp.c_fc.bias");
            w.loadTransposed(b.fcProj.weight(), h + "mlp.c_proj.weight");
            w.load(b.fcProj.bias(), h + "mlp.c_proj.bias");
        }
        w.load(m.norm.weight(), "ln_f.weight");
        w.load(m.norm.bias(), "ln_f.bias");
        return m;
    }

    /** Logits [B, T, vocab] for INT32 tokens [B, T]. */
    @Override
    public Tensor forward(Tensor idx) {
        int t = idx.dim(1);
        int[] pos = new int[t];
        for (int i = 0; i < t; i++) {
            pos[i] = i;
        }
        Tensor x = tokens.forward(idx).add(positions.forward(Tensor.of(pos, t)));
        for (Block b : blocks) {
            x = checkpointing ? org.beehive.jitnn.Grad.checkpoint(in -> b.forward(in[0]), x) : b.forward(x);
        }
        return F.linear(norm.forward(x), tokens.weight(), null);
    }

    static final class Block extends Module {

        private final LayerNorm ln1;
        private final Linear qkv;
        private final Linear proj;
        private final LayerNorm ln2;
        private final Linear fc;
        private final Linear fcProj;
        private final int heads;

        Block(int dim, int heads, Seeds seeds) {
            this.heads = heads;
            this.ln1 = child(new LayerNorm(dim));
            this.qkv = child(new Linear(dim, 3 * dim, seeds.next()));
            this.proj = child(new Linear(dim, dim, seeds.next()));
            this.ln2 = child(new LayerNorm(dim));
            this.fc = child(new Linear(dim, 4 * dim, seeds.next()));
            this.fcProj = child(new Linear(4 * dim, dim, seeds.next()));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = F.multiHeadAttention(ln1.forward(x), qkv.weight(), qkv.bias(), proj.weight(), proj.bias(), heads, null, true);
            x = x.add(h);
            return x.add(fcProj.forward(fc.forward(ln2.forward(x)).geluTanh()));
        }
    }
}
