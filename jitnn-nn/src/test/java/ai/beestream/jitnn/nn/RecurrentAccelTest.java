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
package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.TextCorpus;
import ai.beestream.jitnn.nn.models.CharRnn;
import ai.beestream.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Recurrent gates on the byte-level corpus ({@code -Djitnn.text=<dir>} with the PyTorch
 * references of scripts/pytorch/generative_recurrent.py): two-layer LSTM and GRU language models, time-first
 * batches, Adam with gradient clipping. Same initial weights and batches as PyTorch's cuDNN
 * layers; checks the step-0 gradients against FP64, the loss curve and the evaluation loss.
 */
class RecurrentAccelTest {

    // The same constants as generative_recurrent.py.
    static final int BATCH = 32, SEQ = 64, DIM = 64, HIDDEN = 256, LAYERS = 2, STEPS = 300;
    static final float LR = 2e-3f, CLIP = 1.0f;

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    @Test
    void lstmMatchesPyTorch() throws Exception {
        run("LSTM", "lstm", false);
    }

    @Test
    void gruMatchesPyTorch() throws Exception {
        run("GRU", "gru", true);
    }

    /** [B, T] row-major to [T, B]. */
    private static int[] timeFirst(int[] rows) {
        int[] t = new int[rows.length];
        for (int r = 0; r < BATCH; r++) {
            for (int s = 0; s < SEQ; s++) {
                t[s * BATCH + r] = rows[r * SEQ + s];
            }
        }
        return t;
    }

    private static Tensor loss(CharRnn model, int[] x, int[] y) {
        Tensor logits = model.forward(Tensor.of(timeFirst(x), SEQ, BATCH));
        return F.crossEntropy(logits.reshape(SEQ * BATCH, TextCorpus.VOCAB), Tensor.of(timeFirst(y), SEQ * BATCH));
    }

    private static void run(String name, String kind, boolean gru) throws Exception {
        String d = System.getProperty("jitnn.text", "");
        assumeTrue(!d.isEmpty() && Files.exists(Path.of(d, "corpus.txt")), "set -Djitnn.text=<dir with corpus.txt>");
        Path reference = Path.of(d, kind + "_pytorch.f32");
        Curves torch = Curves.load(reference, 1, "scripts/pytorch/generative_recurrent.py train " + kind);
        TextCorpus corpus = TextCorpus.load(Path.of(d, "corpus.txt"));
        CharRnn model = new CharRnn(TextCorpus.VOCAB, DIM, HIDDEN, LAYERS, gru, 1);
        Adam opt = new Adam(model.parameters(), LR);
        int[] x = new int[BATCH * SEQ];
        int[] y = new int[BATCH * SEQ];
        List<Float> losses = new ArrayList<>();
        long t0 = System.nanoTime();
        long first = 0;
        for (int b = 0; b < STEPS; b++) {
            corpus.trainBatch(b, BATCH, SEQ, x, y);
            Tensor loss = loss(model, x, y);
            opt.zeroGrad();
            loss.backward();
            if (b == 0) {
                Gates.checkGradients(name, model.parameters(), reference.toString());
            }
            opt.clipGradNorm(CLIP);
            opt.step();
            losses.add(loss.item());
            if (b == 0) {
                first = System.nanoTime() - t0;
            }
        }
        double steady = (System.nanoTime() - t0 - first) / 1e6 / (STEPS - 1);
        System.out.printf("[jitnn] %s: %d params, first step %.1f s, then %.1f ms/step (%.0f tokens/s)%n", name, model.parameterCount(), first / 1e9, steady, BATCH * SEQ * 1000 / steady);
        model.train(false);
        int windows = corpus.evalWindows(SEQ);
        double eval = 0;
        int batches = 0;
        for (int w0 = 0; w0 + BATCH <= windows; w0 += BATCH) {
            for (int r = 0; r < BATCH; r++) {
                corpus.evalWindow(w0 + r, SEQ, x, y, r * SEQ);
            }
            eval += Grad.inference(() -> loss(model, x, y)).item();
            batches++;
        }
        eval /= batches;
        System.out.printf("[jitnn] %s eval loss %.4f vs PyTorch %.4f%n", name, eval, torch.metrics()[0]);
        Curves.check(name, Curves.series(losses, 0, 1), torch.series(0, 1), 50, 1e-5, 0.03);
        assertTrue(Math.abs(eval - torch.metrics()[0]) <= 0.03 * torch.metrics()[0], name + ": eval loss " + eval + " vs " + torch.metrics()[0]);
    }
}
