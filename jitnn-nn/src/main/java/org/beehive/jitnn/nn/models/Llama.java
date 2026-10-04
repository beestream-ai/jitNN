package org.beehive.jitnn.nn.models;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.Embedding;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.RMSNorm;

/**
 * A Llama-style decoder: token embedding, pre-RMSNorm blocks of causal attention with rotary
 * position embedding (rotate-half form) and a SwiGLU feed-forward, a final RMSNorm and an untied
 * output projection; no biases. Seeds are handed out in construction order
 * (scripts/pytorch/transformers.py builds the same model).
 */
public final class Llama extends Module {

    private final Embedding tokens;
    private final Block[] blocks;
    private final RMSNorm norm;
    private final Linear output;

    public Llama(int vocab, int dim, int heads, int layers, int hidden, float ropeBase, long seed) {
        Seeds seeds = new Seeds(seed);
        this.tokens = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, vocab, dim), -1));
        this.blocks = new Block[layers];
        for (int i = 0; i < layers; i++) {
            blocks[i] = child(new Block(dim, heads, hidden, ropeBase, seeds));
        }
        this.norm = child(new RMSNorm(dim, 1e-5f));
        this.output = child(new Linear(dim, vocab, false, seeds.next()));
    }

    /** Logits [B, T, vocab] for INT32 tokens [B, T]. */
    @Override
    public Tensor forward(Tensor idx) {
        Tensor x = tokens.forward(idx);
        for (Block b : blocks) {
            x = b.forward(x);
        }
        return output.forward(norm.forward(x));
    }

    static final class Block extends Module {

        private final RMSNorm attnNorm;
        private final Linear wq;
        private final Linear wk;
        private final Linear wv;
        private final Linear wo;
        private final RMSNorm ffnNorm;
        private final Linear w1;
        private final Linear w3;
        private final Linear w2;
        private final int heads;
        private final float ropeBase;

        Block(int dim, int heads, int hidden, float ropeBase, Seeds seeds) {
            this.heads = heads;
            this.ropeBase = ropeBase;
            this.attnNorm = child(new RMSNorm(dim, 1e-5f));
            this.wq = child(new Linear(dim, dim, false, seeds.next()));
            this.wk = child(new Linear(dim, dim, false, seeds.next()));
            this.wv = child(new Linear(dim, dim, false, seeds.next()));
            this.wo = child(new Linear(dim, dim, false, seeds.next()));
            this.ffnNorm = child(new RMSNorm(dim, 1e-5f));
            this.w1 = child(new Linear(dim, hidden, false, seeds.next()));
            this.w3 = child(new Linear(dim, hidden, false, seeds.next()));
            this.w2 = child(new Linear(hidden, dim, false, seeds.next()));
        }

        @Override
        public Tensor forward(Tensor x) {
            int b = x.dim(0);
            int t = x.dim(1);
            int dim = x.dim(2);
            Tensor n = attnNorm.forward(x);
            // [B, T, H, hd] views of the projections: no permutes around attention
            Tensor q = F.ropeHeads(wq.forward(n).reshape(b, t, heads, -1), ropeBase, 0, false);
            Tensor k = F.ropeHeads(wk.forward(n).reshape(b, t, heads, -1), ropeBase, 0, false);
            Tensor v = wv.forward(n).reshape(b, t, heads, -1);
            Tensor y = F.scaledDotProductAttentionBTHD(q, k, v, null, true, -1.0f);
            x = x.add(wo.forward(y.reshape(b, t, dim)));
            Tensor m = ffnNorm.forward(x);
            return x.add(w2.forward(F.swiglu(w1.forward(m), w3.forward(m))));
        }
    }
}
