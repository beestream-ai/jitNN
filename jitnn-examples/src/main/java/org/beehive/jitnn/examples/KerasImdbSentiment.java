package org.beehive.jitnn.examples;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.Conv1d;
import org.beehive.jitnn.nn.Dropout;
import org.beehive.jitnn.nn.Embedding;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.optim.Adam;

/**
 * Tells positive film reviews from negative ones, after Keras' "text classification from scratch":
 * 25,000 IMDB reviews for training and 25,000 for the test (Maas et al., 2011). Words are mapped to
 * the 20,000 most frequent, a learned embedding feeds two strided 1-D convolutions, global max
 * pooling and a dense layer. Downloads the dataset (fast.ai's copy, 144 MB) on first use.
 *
 * <pre>run.sh KerasImdbSentiment [epochs]</pre>
 */
public final class KerasImdbSentiment {

    static final int WORDS = 20000;
    static final int LENGTH = 400;

    /** Embedding 128, dropout, Conv1D(128, 7, stride 3) twice, global max pooling, dense 128, dropout, one logit. */
    static final class Net extends Module {

        final Embedding embedding = child(new Embedding(WORDS, 128, 1));
        final Dropout drop1 = child(new Dropout(0.5f, 2));
        final Conv1d conv1 = child(new Conv1d(128, 128, 7, 3, 0, 1, true, 3));
        final Conv1d conv2 = child(new Conv1d(128, 128, 7, 3, 0, 1, true, 4));
        final Linear dense = child(new Linear(128, 128, 5));
        final Dropout drop2 = child(new Dropout(0.5f, 6));
        final Linear out = child(new Linear(128, 1, 7));

        @Override
        public Tensor forward(Tensor tokens) {
            Tensor x = drop1.forward(embedding.forward(tokens)).permute(0, 2, 1);   // [B, 128, T]
            x = conv2.forward(conv1.forward(x).relu()).relu();
            x = x.max(2, false);                                                      // [B, 128]
            return out.forward(drop2.forward(dense.forwardRelu(x)));
        }
    }

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 3);
        int batch = 64;
        Path dir = imdb();
        List<String> trainText = new ArrayList<>();
        List<Integer> trainY = new ArrayList<>();
        List<String> testText = new ArrayList<>();
        List<Integer> testY = new ArrayList<>();
        read(dir.resolve("train"), trainText, trainY);
        read(dir.resolve("test"), testText, testY);

        Map<String, Integer> vocab = vocabulary(trainText);
        int[] order = Examples.permutation(trainText.size(), 1);
        int[] trainIds = new int[trainText.size() * LENGTH];
        float[] trainLabels = new float[trainText.size()];
        for (int i = 0; i < order.length; i++) {
            encode(trainText.get(order[i]), vocab, trainIds, i);
            trainLabels[i] = trainY.get(order[i]);
        }
        int[] testIds = new int[testText.size() * LENGTH];
        for (int i = 0; i < testText.size(); i++) {
            encode(testText.get(i), vocab, testIds, i);
        }
        System.out.printf("%,d training reviews, %,d test reviews, %,d words kept%n", trainText.size(), testText.size(), WORDS);

        Net net = new Net();
        Adam opt = new Adam(net.parameters(), 1e-3f);
        int steps = trainText.size() / batch;
        for (int epoch = 0; epoch < epochs; epoch++) {
            long t0 = System.nanoTime();
            net.train(true);
            float sum = 0;
            for (int s = 0; s < steps; s++) {
                Tensor ids = Tensor.of(java.util.Arrays.copyOfRange(trainIds, s * batch * LENGTH, (s + 1) * batch * LENGTH), batch, LENGTH);
                Tensor y = Tensor.of(java.util.Arrays.copyOfRange(trainLabels, s * batch, (s + 1) * batch), batch, 1);
                Tensor loss = F.binaryCrossEntropyWithLogits(net.forward(ids), y);
                loss.backward();
                opt.step();
                sum += loss.item();
                opt.zeroGrad();
            }
            System.out.printf("epoch %d: loss %.4f, test accuracy %.2f%%, %.1f s%n", epoch + 1, sum / steps, 100 * accuracy(net, testIds, testY), Examples.seconds(t0));
        }

        String[] mine = { "One of the best films I have seen in years, the acting is superb and the story kept me hooked.",
                "A boring, predictable mess. I walked out after an hour and want my money back." };
        int[] ids = new int[mine.length * LENGTH];
        for (int i = 0; i < mine.length; i++) {
            encode(mine[i], vocab, ids, i);
        }
        float[] logits = Grad.inference(() -> net.forward(Tensor.of(ids, mine.length, LENGTH))).toFloatArray();
        for (int i = 0; i < mine.length; i++) {
            System.out.printf("%.0f%% positive: %s%n", 100 / (1 + Math.exp(-logits[i])), mine[i]);
        }
    }

    static float accuracy(Net net, int[] ids, List<Integer> labels) {
        net.train(false);
        int batch = 500;
        int correct = 0;
        for (int at = 0; at + batch <= labels.size(); at += batch) {
            int from = at;
            float[] logits = Grad.inference(() -> net.forward(Tensor.of(java.util.Arrays.copyOfRange(ids, from * LENGTH, (from + batch) * LENGTH), batch, LENGTH))).toFloatArray();
            for (int i = 0; i < batch; i++) {
                correct += (logits[i] > 0 ? 1 : 0) == labels.get(at + i) ? 1 : 0;
            }
        }
        return (float) correct / labels.size();
    }

    static String[] words(String review) {
        return review.toLowerCase().replace("<br />", " ").split("[^a-z0-9']+");
    }

    /** The {@code WORDS - 2} most frequent training words get ids 2, 3, ...; 0 pads, 1 is any other word. */
    static Map<String, Integer> vocabulary(List<String> texts) {
        Map<String, Integer> counts = new HashMap<>();
        for (String t : texts) {
            for (String w : words(t)) {
                if (!w.isEmpty()) {
                    counts.merge(w, 1, Integer::sum);
                }
            }
        }
        List<Map.Entry<String, Integer>> byCount = new ArrayList<>(counts.entrySet());
        byCount.sort((a, b) -> b.getValue() - a.getValue());
        Map<String, Integer> vocab = new HashMap<>();
        for (int i = 0; i < Math.min(WORDS - 2, byCount.size()); i++) {
            vocab.put(byCount.get(i).getKey(), i + 2);
        }
        return vocab;
    }

    /** The first {@code LENGTH} words of {@code text} as ids into row {@code row} of {@code out}, padded with 0. */
    static void encode(String text, Map<String, Integer> vocab, int[] out, int row) {
        int t = 0;
        for (String w : words(text)) {
            if (!w.isEmpty() && t < LENGTH) {
                out[row * LENGTH + t++] = vocab.getOrDefault(w, 1);
            }
        }
    }

    static void read(Path split, List<String> texts, List<Integer> labels) throws IOException {
        for (String polarity : new String[] { "neg", "pos" }) {
            try (Stream<Path> files = Files.list(split.resolve(polarity))) {
                for (Path f : files.sorted().toList()) {
                    texts.add(Files.readString(f));
                    labels.add(polarity.equals("pos") ? 1 : 0);
                }
            }
        }
    }

    /** The reviews, downloaded and unpacked once (without the unlabelled ones). */
    static Path imdb() throws IOException, InterruptedException {
        Path dir = Examples.dataDir().resolve("imdb");
        if (!Files.isDirectory(dir.resolve("imdb").resolve("test").resolve("pos"))) {
            Path tgz = Examples.download("https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz", dir.resolve("imdb.tgz"));
            Examples.untar(tgz, dir, "imdb/train/pos", "imdb/train/neg", "imdb/test/pos", "imdb/test/neg");
        }
        return dir.resolve("imdb");
    }
}
