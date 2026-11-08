package ai.beestream.jitnn.examples;

import java.nio.charset.StandardCharsets;
import java.util.Random;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.TextCorpus;
import ai.beestream.jitnn.nn.Embedding;
import ai.beestream.jitnn.nn.LSTM;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.optim.Adam;

/**
 * Writes Shakespeare one character at a time, after DL4J's LSTMCharModellingExample and Andrej
 * Karpathy's char-rnn: a two-layer LSTM over bytes learns Tiny Shakespeare, then samples new
 * text, feeding each character back in with the LSTM's state.
 *
 * <pre>run.sh CharLstmShakespeare [steps]</pre>
 */
public final class CharLstmShakespeare {

    /** Byte embedding, a stack of LSTM layers and a linear head; time first, tokens [T, B]. */
    static final class CharModel extends Module {

        final Embedding embedding;
        final LSTM lstm;
        final Linear head;

        CharModel(int dim, int hidden, int layers) {
            this.embedding = child(new Embedding(TextCorpus.VOCAB, dim, 1));
            this.lstm = child(new LSTM(dim, hidden, layers, 2));
            this.head = child(new Linear(hidden, TextCorpus.VOCAB, 3));
        }

        @Override
        public Tensor forward(Tensor tokens) {
            return head.forward(lstm.forward(embedding.forward(tokens)));
        }

        /** One step for one sequence from states h, c: the logits [1, 1, vocab] and the next states. */
        LSTM.Output step(int token, Tensor h, Tensor c) {
            LSTM.Output o = lstm.forward(embedding.forward(Tensor.of(new int[] { token }, 1, 1)), h, c, null);
            return new LSTM.Output(head.forward(o.y()), o.h(), o.c());
        }
    }

    public static void main(String[] args) throws Exception {
        int steps = Examples.intArg(args, 0, 2000);
        int rows = 32;
        int seq = 100;
        int layers = 2;
        int hidden = 256;
        TextCorpus text = TextCorpus.load(Examples.shakespeare());
        CharModel model = new CharModel(64, hidden, layers);
        Adam opt = new Adam(model.parameters(), 2e-3f);
        opt.clipGradNorm(5.0f);
        System.out.printf("char LSTM with %,d parameters, %,d characters of text%n", model.parameterCount(), text.length());

        int[] inputs = new int[rows * seq];
        int[] targets = new int[rows * seq];
        int[] inT = new int[rows * seq];
        int[] outT = new int[rows * seq];
        long t0 = System.nanoTime();
        float sum = 0;
        for (int step = 1; step <= steps; step++) {
            text.trainBatch(step, rows, seq, inputs, targets);
            for (int r = 0; r < rows; r++) {          // [rows, seq] to time first [seq, rows]
                for (int t = 0; t < seq; t++) {
                    inT[t * rows + r] = inputs[r * seq + t];
                    outT[t * rows + r] = targets[r * seq + t];
                }
            }
            Tensor logits = model.forward(Tensor.of(inT, seq, rows));
            Tensor loss = F.crossEntropy(logits.reshape(seq * rows, TextCorpus.VOCAB), Tensor.of(outT, seq * rows));
            loss.backward();
            opt.step();
            sum += loss.item();
            opt.zeroGrad();
            if (step % 200 == 0) {
                System.out.printf("step %d: loss %.3f, %.1f s%n", step, sum / 200, Examples.seconds(t0));
                sum = 0;
            }
        }
        System.out.println();
        System.out.println(generate(model, "ROMEO:\n", 600, 0.8f, layers, hidden));
    }

    /** Samples {@code length} characters after {@code prompt} at the given temperature. */
    static String generate(CharModel model, String prompt, int length, float temperature, int layers, int hidden) {
        model.train(false);
        Random random = new Random(11);
        byte[] start = prompt.getBytes(StandardCharsets.UTF_8);
        byte[] out = new byte[start.length + length];
        System.arraycopy(start, 0, out, 0, start.length);
        float[] h = new float[layers * hidden];
        float[] c = new float[layers * hidden];
        float[] logits = null;
        try (var noGrad = Grad.disabled()) {
            for (int i = 0; i < out.length - 1; i++) {
                // every step has the same shapes, so one compiled plan serves the whole text
                LSTM.Output o = model.step(out[i] & 0xFF, Tensor.of(h, layers, 1, hidden), Tensor.of(c, layers, 1, hidden));
                logits = o.y().toFloatArray();
                h = o.h().toFloatArray();
                c = o.c().toFloatArray();
                if (i + 1 >= start.length) {
                    out[i + 1] = (byte) Examples.sample(Examples.softmax(logits, 0, TextCorpus.VOCAB, temperature), random);
                }
            }
        }
        return new String(out, StandardCharsets.UTF_8);
    }
}
