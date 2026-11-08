package org.beehive.jitnn.examples;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.Dropout;
import org.beehive.jitnn.nn.LayerNorm;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.optim.Adam;
import org.beehive.jitnn.nn.optim.LRScheduler;
import org.beehive.jitnn.ops.AttentionSoftmax;

/**
 * Trains a GPT from scratch on Tiny Shakespeare, Andrej Karpathy's nanoGPT with its
 * shakespeare_char settings: the 65 characters of the text as tokens, 6 layers, 6 heads, 384 wide,
 * 256 tokens of context, dropout 0.2 (on the embeddings, the attention weights and both residual
 * branches), no biases, batches of 64 random windows, AdamW (lr 1e-3,
 * beta2 0.99, weight decay 0.1 on the matrices only) with 100 warm-up steps and a cosine decay to
 * 1e-4 over 5,000 steps, gradients clipped to 1. Every 250 steps the loss is estimated on 200
 * training and 200 validation batches, as nanoGPT does. Afterwards it writes a new scene.
 *
 * <pre>run.sh NanoGptShakespeare [steps]</pre>
 */
public final class NanoGptShakespeare {

    static final int LAYERS = 6;
    static final int HEADS = 6;
    static final int DIM = 384;
    static final int CONTEXT = 256;
    static final float DROPOUT = 0.2f;

    /** A linear layer without bias, initialised N(0, std) as nanoGPT does. */
    static final class Proj extends Module {

        final Tensor weight;

        Proj(int in, int out, float std, long seed) {
            this.weight = parameter(Tensor.normal(seed, 0, std, out, in));
        }

        @Override
        public Tensor forward(Tensor x) {
            return F.linear(x, weight, null);
        }
    }

    /** Pre-LayerNorm block: causal self-attention and a GELU MLP, each with residual dropout. */
    static final class Block extends Module {

        final LayerNorm ln1 = child(new LayerNorm(DIM, 1e-5f, false));
        final LayerNorm ln2 = child(new LayerNorm(DIM, 1e-5f, false));
        final Proj qkv;
        final Proj proj;
        final Proj fc;
        final Proj fcProj;
        final Dropout attnDrop;
        final Dropout drop1;
        final Dropout drop2;

        Block(long seed) {
            float residual = 0.02f / (float) Math.sqrt(2 * LAYERS);      // nanoGPT's scaled init of the projections into the residual
            this.qkv = child(new Proj(DIM, 3 * DIM, 0.02f, seed));
            this.proj = child(new Proj(DIM, DIM, residual, seed + 1));
            this.fc = child(new Proj(DIM, 4 * DIM, 0.02f, seed + 2));
            this.fcProj = child(new Proj(4 * DIM, DIM, residual, seed + 3));
            this.attnDrop = child(new Dropout(DROPOUT, seed + 6));
            this.drop1 = child(new Dropout(DROPOUT, seed + 4));
            this.drop2 = child(new Dropout(DROPOUT, seed + 5));
        }

        @Override
        public Tensor forward(Tensor x) {
            int b = x.dim(0);
            int t = x.dim(1);
            Tensor[] qkvs = qkv.forward(ln1.forward(x)).reshape(b, t, 3, HEADS, DIM / HEADS).permute(2, 0, 3, 1, 4).chunk(3, 0);
            Tensor q = qkvs[0].reshape(b, HEADS, t, DIM / HEADS);
            Tensor k = qkvs[1].reshape(b, HEADS, t, DIM / HEADS);
            Tensor v = qkvs[2].reshape(b, HEADS, t, DIM / HEADS);
            Tensor a = attention(q, k, v).permute(0, 2, 1, 3).reshape(b, t, DIM);
            x = x.add(drop1.forward(proj.forward(a)));
            return x.add(drop2.forward(fcProj.forward(fc.forward(ln2.forward(x)).gelu())));
        }

        /** Causal attention; in training with dropout on the attention weights, as nanoGPT's SDPA dropout_p. */
        Tensor attention(Tensor q, Tensor k, Tensor v) {
            if (!isTraining()) {
                return F.scaledDotProductAttention(q, k, v, null, true, -1);
            }
            Tensor weights = AttentionSoftmax.apply(q.matmul(k, false, true), (float) (1.0 / Math.sqrt(q.dim(-1))), true);
            return attnDrop.forward(weights).matmul(v);
        }
    }

    /** Token and position embeddings, the blocks, a final LayerNorm, logits from the tied token embedding. */
    static final class Gpt extends Module {

        final Tensor wte;
        final Tensor wpe;
        final Dropout drop;
        final List<Block> blocks = new ArrayList<>();
        final LayerNorm lnf = child(new LayerNorm(DIM, 1e-5f, false));

        Gpt(int vocab) {
            this.wte = parameter(Tensor.normal(1, 0, 0.02f, vocab, DIM));
            this.wpe = parameter(Tensor.normal(2, 0, 0.02f, CONTEXT, DIM));
            this.drop = child(new Dropout(DROPOUT, 3));
            for (int i = 0; i < LAYERS; i++) {
                blocks.add(child(new Block(100 + 10L * i)));
            }
        }

        @Override
        public Tensor forward(Tensor idx) {
            int t = idx.dim(1);
            int[] pos = new int[t];
            for (int i = 0; i < t; i++) {
                pos[i] = i;
            }
            Tensor x = drop.forward(F.embedding(idx, wte).add(F.embedding(Tensor.of(pos, t), wpe)));
            for (Block b : blocks) {
                x = b.forward(x);
            }
            return F.linear(lnf.forward(x), wte, null);
        }
    }

    public static void main(String[] args) throws Exception {
        int steps = Examples.intArg(args, 0, 5000);
        int batch = 64;
        String text = Files.readString(Examples.shakespeare());
        TreeSet<Character> chars = new TreeSet<>();
        for (char c : text.toCharArray()) {
            chars.add(c);
        }
        List<Character> alphabet = new ArrayList<>(chars);
        int vocab = alphabet.size();
        int[] tokens = new int[text.length()];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = alphabet.indexOf(text.charAt(i));
        }
        int split = tokens.length * 9 / 10;

        Gpt gpt = new Gpt(vocab);
        // weight decay on the matrices (2-D parameters) only, as nanoGPT's configure_optimizers
        List<Tensor> decay = new ArrayList<>();
        List<Tensor> noDecay = new ArrayList<>();
        for (Tensor p : gpt.parameters()) {
            (p.rank() >= 2 ? decay : noDecay).add(p);
        }
        Adam opt = new Adam(decay, 1e-3f, 0.9f, 0.99f, 1e-8f, 0.1f, true);
        Adam optNoDecay = new Adam(noDecay, 1e-3f, 0.9f, 0.99f, 1e-8f, 0, true);
        LRScheduler schedule = LRScheduler.warmupCosine(opt, 100, steps, 0.1f);
        LRScheduler scheduleNoDecay = LRScheduler.warmupCosine(optNoDecay, 100, steps, 0.1f);
        System.out.printf("GPT with %,d parameters, %d characters in the vocabulary, %,d characters of text%n", gpt.parameterCount(), vocab, tokens.length);

        Random random = new Random(1337);
        int[] x = new int[batch * CONTEXT];
        int[] y = new int[batch * CONTEXT];
        long t0 = System.nanoTime();
        for (int step = 0; step <= steps; step++) {
            if (step % 250 == 0) {
                float[] losses = estimate(gpt, tokens, split, random, batch);
                System.out.printf("step %d: train loss %.4f, val loss %.4f, %.1f s%n", step, losses[0], losses[1], Examples.seconds(t0));
            }
            if (step == steps) {
                break;
            }
            gpt.train(true);
            sample(tokens, 0, split, random, batch, x, y);
            Tensor logits = gpt.forward(Tensor.of(x, batch, CONTEXT));
            Tensor loss = F.crossEntropy(logits.reshape(batch * CONTEXT, vocab), Tensor.of(y, batch * CONTEXT));
            loss.backward();
            // every step, as nanoGPT's grad_clip (here the norm of each parameter group)
            opt.clipGradNorm(1.0f);
            optNoDecay.clipGradNorm(1.0f);
            opt.step();
            optNoDecay.step();
            loss.item();
            opt.zeroGrad();
            optNoDecay.zeroGrad();
            schedule.step();
            scheduleNoDecay.step();
        }
        System.out.printf("%,d steps in %.1f s%n%n", steps, Examples.seconds(t0));
        System.out.println(generate(gpt, alphabet, 500));
    }

    /** nanoGPT's estimate_loss: the mean loss over 200 random batches of each split, without dropout. */
    static float[] estimate(Gpt gpt, int[] tokens, int split, Random random, int batch) {
        gpt.train(false);
        int[] x = new int[batch * CONTEXT];
        int[] y = new int[batch * CONTEXT];
        float[] out = new float[2];
        for (int s = 0; s < 2; s++) {
            float sum = 0;
            for (int k = 0; k < 200; k++) {
                if (s == 0) {
                    sample(tokens, 0, split, random, batch, x, y);
                } else {
                    sample(tokens, split, tokens.length, random, batch, x, y);
                }
                int vocab = gpt.wte.dim(0);
                sum += F.crossEntropy(Grad.inference(() -> gpt.forward(Tensor.of(x, batch, CONTEXT))).reshape(batch * CONTEXT, vocab), Tensor.of(y, batch * CONTEXT)).item();
            }
            out[s] = sum / 200;
        }
        return out;
    }

    /** {@code batch} random windows of [from, to): inputs and the next characters. */
    static void sample(int[] tokens, int from, int to, Random random, int batch, int[] x, int[] y) {
        for (int r = 0; r < batch; r++) {
            int start = from + random.nextInt(to - from - CONTEXT - 1);
            System.arraycopy(tokens, start, x, r * CONTEXT, CONTEXT);
            System.arraycopy(tokens, start + 1, y, r * CONTEXT, CONTEXT);
        }
    }

    /** Samples characters from a fixed window, so one compiled plan serves every step. */
    static String generate(Gpt gpt, List<Character> alphabet, int length) {
        gpt.train(false);
        Random random = new Random(3);
        int vocab = alphabet.size();
        List<Integer> text = new ArrayList<>(List.of(alphabet.indexOf('\n')));
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < length; i++) {
            int from = Math.max(0, text.size() - CONTEXT);
            int len = text.size() - from;
            int[] window = new int[CONTEXT];
            for (int j = 0; j < len; j++) {
                window[j] = text.get(from + j);
            }
            Tensor row = Tensor.of(new int[] { len - 1 }, 1);
            float[] logits = Grad.inference(() -> F.embedding(row, gpt.forward(Tensor.of(window, 1, CONTEXT)).reshape(CONTEXT, vocab))).toFloatArray();
            int next = Examples.sample(Examples.softmax(logits, 0, vocab, 0.8f), random);
            text.add(next);
            out.append(alphabet.get(next));
        }
        return out.toString();
    }
}
