package org.beehive.jitnn.nn.models;

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
            x = b.forward(x);
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
