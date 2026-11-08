package org.beehive.jitnn.examples;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.models.GPT2;
import org.beehive.jitnn.nn.text.Tokenizer;

/**
 * Continues a prompt with GPT-2 (124M) from Hugging Face, after DJL's TextGeneration example.
 * The weights (548 MB) and tokenizer are downloaded on first use. Every step runs the model on a
 * fixed window of 128 tokens, so one compiled plan serves the whole generation; the logits of the
 * last real token are gathered on the device and sampled on the host (top 40, temperature 0.8).
 *
 * <pre>run.sh Gpt2TextGeneration ["prompt"] [tokens]</pre>
 */
public final class Gpt2TextGeneration {

    public static void main(String[] args) throws Exception {
        String prompt = args.length > 0 ? args[0] : "DeepMind Company is";
        int tokens = Examples.intArg(args, 1, 60);
        int window = 128;
        Path dir = Examples.huggingFace("gpt2", "config.json", "model.safetensors", "tokenizer.json");
        GPT2 gpt2 = GPT2.fromHuggingFace(dir);
        gpt2.train(false);
        Tokenizer tok = Tokenizer.load(dir);
        int vocab = tok.vocabSize();
        int eos = tok.id("<|endoftext|>");

        int[] text = tok.tokenize(prompt);
        Random random = new Random(42);
        System.out.print(prompt);
        long t0 = System.nanoTime();
        double first = 0;
        for (int i = 0; i < tokens; i++) {
            // the last `window` tokens, padded at the end; causal attention never looks at the padding
            int from = Math.max(0, text.length - window);
            int len = text.length - from;
            int[] ids = new int[window];
            Arrays.fill(ids, eos);
            System.arraycopy(text, from, ids, 0, len);
            Tensor row = Tensor.of(new int[] { len - 1 }, 1);
            float[] logits = Grad.inference(() -> F.embedding(row, gpt2.forward(Tensor.of(ids, 1, window)).reshape(window, vocab))).toFloatArray();
            if (i == 0) {
                first = Examples.seconds(t0);
            }
            int next = topK(logits, 40, 0.8f, random);
            text = Arrays.copyOf(text, text.length + 1);
            text[text.length - 1] = next;
            System.out.print(tok.decode(new int[] { next }, true));
            System.out.flush();
        }
        double rest = Examples.seconds(t0) - first;
        System.out.printf("%n%nfirst token %.2f s (compiles the plan), then %.1f tokens/s%n", first, (tokens - 1) / rest);
    }

    /** Samples from the {@code k} most likely tokens at the given temperature. */
    static int topK(float[] logits, int k, float temperature, Random random) {
        int[] best = Examples.topK(logits, k);
        float[] picked = new float[k];
        for (int i = 0; i < k; i++) {
            picked[i] = logits[best[i]];
        }
        return best[Examples.sample(Examples.softmax(picked, 0, k, temperature), random)];
    }
}
