/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.nn.models;

import java.nio.file.Path;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Embedding;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.RMSNorm;
import ai.beestream.jitnn.ops.Rope;

/**
 * A Llama-style decoder: token embedding, pre-RMSNorm blocks of causal attention with rotary
 * position embedding (rotate-half form) and a SwiGLU feed-forward, a final RMSNorm and an untied
 * output projection; no biases. Seeds are handed out in construction order
 * (scripts/pytorch/transformers.py builds the same model). With fewer key/value heads than query
 * heads the attention is grouped-query (GQA), as Llama 2 70B and Llama 3; {@link #fromHuggingFace}
 * builds a model from a Hugging Face Llama checkpoint.
 */
public final class Llama extends Module {

    private final Embedding tokens;
    private final Block[] blocks;
    private final RMSNorm norm;
    private final Linear output;
    /** The output projection is the token embedding (tied), when output is null. */

    private boolean checkpointing;

    /**
     * With {@code on}, training keeps no activations inside the blocks: each block runs again in the
     * backward pass ({@link ai.beestream.jitnn.Grad#checkpoint}), for less memory and about a third more compute.
     */
    public Llama checkpointing(boolean on) {
        checkpointing = on;
        return this;
    }

    /** Rescales the RoPE frequencies of every block (Llama 3.1's rope_scaling); null for none. */
    public Llama ropeScaling(Rope.Scaling scaling) {
        for (Block b : blocks) {
            b.ropeScaling = scaling;
        }
        return this;
    }

    /**
     * A decoder over {@code vocab} tokens with {@code layers} blocks of {@code dim} features in
     * {@code heads} heads, a feed-forward of {@code hidden} features and rotary base {@code ropeBase}, initialised
     * from {@code seed}.
     */
    public Llama(int vocab, int dim, int heads, int layers, int hidden, float ropeBase, long seed) {
        this(vocab, dim, heads, heads, layers, hidden, ropeBase, 1e-5f, false, false, seed);
    }

    /**
     * A Llama with {@code kvHeads} key/value heads (grouped-query attention when fewer than
     * {@code heads}), RMSNorm epsilon {@code eps}, the output projection tied to the token
     * embedding when {@code tied}, and biases on the q, k and v projections when {@code qkvBias}.
     */
    public Llama(int vocab, int dim, int heads, int kvHeads, int layers, int hidden, float ropeBase, float eps, boolean tied, boolean qkvBias, long seed) {
        if (heads % kvHeads != 0 || dim % heads != 0) {
            throw new IllegalArgumentException("Llama: " + heads + " heads over " + kvHeads + " key/value heads, dimension " + dim);
        }
        Seeds seeds = new Seeds(seed);
        this.tokens = child(new Embedding(Init.uniformStd(seeds.next(), 0.02f, vocab, dim), -1));
        this.blocks = new Block[layers];
        for (int i = 0; i < layers; i++) {
            blocks[i] = child(new Block(dim, heads, kvHeads, hidden, ropeBase, eps, qkvBias, seeds));
        }
        this.norm = child(new RMSNorm(dim, eps));
        this.output = tied ? null : child(new Linear(dim, vocab, false, seeds.next()));
    }

    /**
     * A Llama from a Hugging Face checkpoint directory (config.json and safetensors, as
     * {@link HfCheckpoint}): model_type llama (or mistral without a sliding window, qwen2 with its
     * q, k, v biases), grouped-query attention, tied or separate output weights, and linear or
     * Llama 3.1 RoPE scaling. A head dimension other than hidden_size / heads is not supported.
     */
    public static Llama fromHuggingFace(Path dir) {
        HfCheckpoint w = HfCheckpoint.open(dir);
        String type = w.configString("model_type", "llama");
        if (!type.equals("llama") && !type.equals("mistral") && !type.equals("qwen2")) {
            throw new IllegalArgumentException("model_type " + type + " is not supported (llama, mistral, qwen2 are)");
        }
        // transformers 5 writes rope_parameters (theta included), earlier versions rope_theta and rope_scaling
        java.util.Map<?, ?> rope = w.config().get("rope_parameters") instanceof java.util.Map<?, ?> p ? p
                : w.config().get("rope_scaling") instanceof java.util.Map<?, ?> q ? q : null;
        float theta = w.config().get("rope_theta") instanceof Number t ? t.floatValue()
                : rope != null && rope.get("rope_theta") instanceof Number u ? u.floatValue() : 10000f;
        Rope.Scaling scaling = null;
        if (rope != null && (rope.containsKey("rope_type") || rope.containsKey("type"))) {
            java.util.Map<?, ?> rs = rope;
            Object kind = rs.containsKey("rope_type") ? rs.get("rope_type") : rs.get("type");
            double factor = rs.get("factor") instanceof Number f ? f.doubleValue() : 1.0;
            scaling = switch (String.valueOf(kind)) {
                case "default" -> null;
                case "linear" -> Rope.Scaling.linear(factor);
                case "llama3" -> Rope.Scaling.llama3(factor, ((Number) rs.get("low_freq_factor")).doubleValue(), ((Number) rs.get("high_freq_factor")).doubleValue(),
                        ((Number) rs.get("original_max_position_embeddings")).intValue());
                default -> throw new IllegalArgumentException("rope_scaling " + kind + " is not supported (linear, llama3 are)");
            };
        }
        if (w.config().get("sliding_window") != null && !type.equals("qwen2") || Boolean.TRUE.equals(w.config().get("use_sliding_window"))) {
            throw new IllegalArgumentException("sliding-window attention is not supported");
        }
        if (!w.configString("hidden_act", "silu").equals("silu")) {
            throw new IllegalArgumentException("activation " + w.configString("hidden_act", "silu") + " is not supported (silu is)");
        }
        int dim = w.configInt("hidden_size");
        int heads = w.configInt("num_attention_heads");
        int kvHeads = w.config().containsKey("num_key_value_heads") ? w.configInt("num_key_value_heads") : heads;
        if (w.config().get("head_dim") != null && w.configInt("head_dim") != dim / heads) {
            throw new IllegalArgumentException("head_dim " + w.configInt("head_dim") + " other than hidden_size / heads is not supported");
        }
        boolean tied = Boolean.TRUE.equals(w.config().get("tie_word_embeddings")) || !w.has("lm_head.weight");
        boolean bias = type.equals("qwen2") || Boolean.TRUE.equals(w.config().get("attention_bias"));
        Llama m = new Llama(w.configInt("vocab_size"), dim, heads, kvHeads, w.configInt("num_hidden_layers"), w.configInt("intermediate_size"),
                theta, w.configFloat("rms_norm_eps", 1e-6f), tied, bias, 0);
        m.ropeScaling(scaling);
        w.load(m.tokens.weight(), "model.embed_tokens.weight");
        for (int i = 0; i < m.blocks.length; i++) {
            Block b = m.blocks[i];
            String l = "model.layers." + i + ".";
            w.load(b.attnNorm.weight(), l + "input_layernorm.weight");
            w.load(b.wq.weight(), l + "self_attn.q_proj.weight");
            w.load(b.wk.weight(), l + "self_attn.k_proj.weight");
            w.load(b.wv.weight(), l + "self_attn.v_proj.weight");
            if (bias) {
                w.load(b.wq.bias(), l + "self_attn.q_proj.bias");
                w.load(b.wk.bias(), l + "self_attn.k_proj.bias");
                w.load(b.wv.bias(), l + "self_attn.v_proj.bias");
            }
            w.load(b.wo.weight(), l + "self_attn.o_proj.weight");
            w.load(b.ffnNorm.weight(), l + "post_attention_layernorm.weight");
            w.load(b.w1.weight(), l + "mlp.gate_proj.weight");
            w.load(b.w3.weight(), l + "mlp.up_proj.weight");
            w.load(b.w2.weight(), l + "mlp.down_proj.weight");
        }
        w.load(m.norm.weight(), "model.norm.weight");
        if (m.output != null) {
            w.load(m.output.weight(), "lm_head.weight");
        }
        return m;
    }

    /** Logits [B, T, vocab] for INT32 tokens [B, T]. */
    @Override
    public Tensor forward(Tensor idx) {
        Tensor x = tokens.forward(idx);
        for (Block b : blocks) {
            x = checkpointing ? ai.beestream.jitnn.Grad.checkpoint(in -> b.forward(in[0]), x) : b.forward(x);
        }
        Tensor h = norm.forward(x);
        return output != null ? output.forward(h) : F.linear(h, tokens.weight(), null);
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
        private final int kvHeads;
        private final float ropeBase;
        private Rope.Scaling ropeScaling;

        Block(int dim, int heads, int kvHeads, int hidden, float ropeBase, float eps, boolean qkvBias, Seeds seeds) {
            this.heads = heads;
            this.kvHeads = kvHeads;
            this.ropeBase = ropeBase;
            int kvDim = dim / heads * kvHeads;
            this.attnNorm = child(new RMSNorm(dim, eps));
            this.wq = child(new Linear(dim, dim, qkvBias, seeds.next()));
            this.wk = child(new Linear(dim, kvDim, qkvBias, seeds.next()));
            this.wv = child(new Linear(dim, kvDim, qkvBias, seeds.next()));
            this.wo = child(new Linear(dim, dim, false, seeds.next()));
            this.ffnNorm = child(new RMSNorm(dim, eps));
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
            Tensor q = F.ropeHeads(wq.forward(n).reshape(b, t, heads, -1), ropeBase, ropeScaling, 0, false);
            Tensor k = F.ropeHeads(wk.forward(n).reshape(b, t, kvHeads, -1), ropeBase, ropeScaling, 0, false);
            Tensor v = wv.forward(n).reshape(b, t, kvHeads, -1);
            Tensor y = F.scaledDotProductAttentionBTHD(q, k, v, null, true, -1.0f);
            x = x.add(wo.forward(y.reshape(b, t, dim)));
            Tensor m = ffnNorm.forward(x);
            return x.add(w2.forward(F.swiglu(w1.forward(m), w3.forward(m))));
        }
    }
}
