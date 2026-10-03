package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.TextCorpus;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.models.Bert;
import org.beehive.jitnn.nn.models.GPT2;
import org.beehive.jitnn.nn.models.Llama;
import org.beehive.jitnn.nn.optim.Adam;
import org.beehive.jitnn.nn.optim.LRScheduler;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Transformer gates against PyTorch (scripts/pytorch/transformers.py), on a byte-level corpus
 * ({@code -Djitnn.text=<dir>} holding corpus.txt and the PyTorch references): GPT-2 and a
 * Llama-style decoder as language models, and BERT with masked language modelling and padding
 * masks. Same initial weights and batches; AdamW with warm-up and cosine decay, gradient
 * clipping at norm 1. Checks the step-0 gradients against PyTorch's FP64 ones, the step-0
 * loss, the end of the loss curve and the evaluation loss.
 */
class TransformerAccelTest {

    // The same constants as transformers.py.
    static final int BATCH = 16;
    static final int SEQ = 128;
    static final int DIM = 128;
    static final int HEADS = 4;
    static final int LAYERS = 4;
    static final int LLAMA_HIDDEN = 352;
    static final int MASK = 256;
    static final int PAD = 257;
    static final int BERT_VOCAB = 258;
    static final int PREDICTIONS = 19;
    static final float LR = 1e-3f;
    static final float WEIGHT_DECAY = 0.01f;
    static final int WARMUP = 30;
    static final float MIN_RATIO = 0.1f;
    static final float CLIP = 1.0f;
    static final int STEPS = Integer.getInteger("jitnn.transformer.steps", 300);

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void gpt2MatchesPyTorch() throws Exception {
        run("GPT-2", "gpt2", new GPT2(TextCorpus.VOCAB, SEQ, DIM, HEADS, LAYERS, 1));
    }

    @Test
    void llamaMatchesPyTorch() throws Exception {
        run("Llama", "llama", new Llama(TextCorpus.VOCAB, DIM, HEADS, LAYERS, LLAMA_HIDDEN, 10000.0f, 1));
    }

    @Test
    void bertMatchesPyTorch() throws Exception {
        run("BERT", "bert", new Bert(BERT_VOCAB, SEQ, DIM, HEADS, LAYERS, 1));
    }

    /** One MLM batch: padded, masked inputs, the additive padding mask, the masked flat rows and their labels. */
    record Mlm(int[] inputs, float[] mask, int[] rows, int[] labels) {
    }

    /**
     * As mlm_batch in transformers.py: row r is padded to SEQ - (key_r * 7) % 33 tokens and its
     * {@link #PREDICTIONS} real positions with the smallest integer hash are masked; a fixed count
     * keeps every step's shapes, and so its compiled plan, the same.
     */
    static Mlm mlm(int[] x, long key) {
        int[] in = x.clone();
        float[] mask = new float[BATCH * SEQ];
        int[] rows = new int[BATCH * PREDICTIONS];
        int[] labels = new int[BATCH * PREDICTIONS];
        for (int r = 0; r < BATCH; r++) {
            int length = (int) (SEQ - ((key * BATCH + r) * 7) % 33);
            final int row = r;
            Integer[] order = new Integer[length];
            for (int p = 0; p < length; p++) {
                order[p] = p;
            }
            java.util.Arrays.sort(order, (a, b) -> {
                int c = Long.compare(hash(key, row, a), hash(key, row, b));
                return c != 0 ? c : Integer.compare(a, b);
            });
            int[] chosen = new int[PREDICTIONS];
            for (int k = 0; k < PREDICTIONS; k++) {
                chosen[k] = order[k];
            }
            java.util.Arrays.sort(chosen);
            for (int k = 0; k < PREDICTIONS; k++) {
                int at = r * SEQ + chosen[k];
                rows[r * PREDICTIONS + k] = at;
                labels[r * PREDICTIONS + k] = x[at];
                in[at] = MASK;
            }
            for (int p = length; p < SEQ; p++) {
                in[r * SEQ + p] = PAD;
                mask[r * SEQ + p] = -1e9f;
            }
        }
        return new Mlm(in, mask, rows, labels);
    }

    private static long hash(long key, int r, int p) {
        return (((key * BATCH + r) * SEQ + p) * 2654435761L) & 0xFFFFFFFFL;
    }

    static Tensor loss(String kind, Module model, int[] x, int[] y, long key) {
        if (kind.equals("bert")) {
            Mlm m = mlm(x, key);
            Tensor logits = ((Bert) model).mlmLogits(Tensor.of(m.inputs(), BATCH, SEQ), Tensor.of(m.mask(), BATCH, 1, 1, SEQ), Tensor.of(m.rows(), m.rows().length));
            return F.crossEntropy(logits, Tensor.of(m.labels(), m.labels().length));
        }
        Tensor logits = model.forward(Tensor.of(x, BATCH, SEQ));
        return F.crossEntropy(logits.reshape(BATCH * SEQ, -1), Tensor.of(y, BATCH * SEQ));
    }

    private static void run(String name, String kind, Module model) throws Exception {
        String dir = System.getProperty("jitnn.text", "");
        assumeTrue(!dir.isEmpty() && Files.exists(Path.of(dir, "corpus.txt")), "set -Djitnn.text=<dir with corpus.txt>");
        Path reference = Path.of(dir, kind + "_pytorch.f32");
        assumeTrue(Files.exists(reference), "run scripts/pytorch/transformers.py train " + kind + " to write " + reference);
        TextCorpus corpus = TextCorpus.load(Path.of(dir, "corpus.txt"));
        List<Tensor> params = model.parameters();
        Adam opt = Adam.adamW(params, LR, WEIGHT_DECAY);
        LRScheduler sched = LRScheduler.warmupCosine(opt, WARMUP, STEPS, MIN_RATIO);
        int[] x = new int[BATCH * SEQ];
        int[] y = new int[BATCH * SEQ];
        List<Float> losses = new ArrayList<>();
        long[] plans0 = Engine.planStats();
        long[] split0 = Engine.timeSplit();
        long t0 = System.nanoTime();
        long firstStep = 0;
        for (int b = 0; b < STEPS; b++) {
            corpus.trainBatch(b, BATCH, SEQ, x, y);
            Tensor loss = loss(kind, model, x, y, b);
            loss.backward();
            if (b == 0) {
                Gates.checkGradients(name, params, reference.toString());
            }
            Grad.clipNorm(params, CLIP);
            opt.step();
            sched.step();
            losses.add(loss.item());
            opt.zeroGrad();
            if (b == 0) {
                firstStep = System.nanoTime() - t0;
            }
        }
        double steady = (System.nanoTime() - t0 - firstStep) / 1e6 / (STEPS - 1);
        long[] split1 = Engine.timeSplit();
        long[] plans1 = Engine.planStats();

        model.train(false);
        int windows = corpus.evalWindows(SEQ);
        double evalLoss = 0;
        int batches = 0;
        for (int w0 = 0; w0 + BATCH <= windows; w0 += BATCH) {
            for (int r = 0; r < BATCH; r++) {
                corpus.evalWindow(w0 + r, SEQ, x, y, r * SEQ);
            }
            long key = 1_000_000L + w0;
            evalLoss += Grad.inference(() -> loss(kind, model, x, y, key)).item();
            batches++;
        }
        evalLoss /= batches;

        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(reference)).order(ByteOrder.LITTLE_ENDIAN);
        float torchEval = buf.getFloat();
        float[] torchLosses = new float[buf.remaining() / 4];
        buf.asFloatBuffer().get(torchLosses);
        assertTrue(torchLosses.length == STEPS, name + ": PyTorch ran " + torchLosses.length + " steps, jitNN " + STEPS);
        double gap = 0;
        for (int i = 0; i < 10; i++) {
            gap = Math.max(gap, Math.abs(losses.get(i) - torchLosses[i]) / Math.abs(torchLosses[i]));
        }
        double meanJ = 0;
        double meanT = 0;
        int tail = Math.min(50, STEPS / 4);
        for (int i = STEPS - tail; i < STEPS; i++) {
            meanJ += losses.get(i) / tail;
            meanT += torchLosses[i] / tail;
        }
        System.out.printf("[jitnn] %s per step: jitNN executor %.1f ms, TornadoVM execute %.1f ms%n", name, (split1[0] - split0[0]) / 1e6 / STEPS, (split1[1] - split0[1]) / 1e6 / STEPS);
        System.out.printf("[jitnn] %s: %d params, first step %.1f s, then %.1f ms/step (%.0f tokens/s), %d plans compiled; vs PyTorch: steps 0-9 gap %.2e, "
                + "mean loss of the last %d steps %.4f vs %.4f, eval loss %.4f vs %.4f%n", name, model.parameterCount(), firstStep / 1e9, steady,
                BATCH * SEQ * 1000 / steady, plans1[0] - plans0[0], gap, tail, meanJ, meanT, evalLoss, torchEval);
        assertTrue(Math.abs(losses.get(0) - torchLosses[0]) <= 1e-5 * torchLosses[0], name + ": step-0 loss " + losses.get(0) + " vs " + torchLosses[0]);
        assertTrue(Math.abs(meanJ - meanT) <= 0.05 * meanT, name + ": final mean loss " + meanJ + " vs " + meanT);
        assertTrue(Math.abs(evalLoss - torchEval) <= 0.05 * torchEval, name + ": eval loss " + evalLoss + " vs " + torchEval);
    }
}
