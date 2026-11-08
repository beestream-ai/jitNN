package ai.beestream.jitnn.examples;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.SparseMatrix;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Dropout;
import ai.beestream.jitnn.nn.GraphConv;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.ReLU;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.optim.Adam;

/**
 * Classifies scientific papers by the papers they cite, PyTorch's GCN example with its settings
 * (Kipf and Welling, 2017): Cora's 2,708 papers, their 1,433-word bag-of-words features and 5,429
 * citations; 1,000 random papers for the test, 500 for validation, the rest for training. Two graph
 * convolutions (16 hidden units, dropout 0.5 between them), Adam (lr 0.01, weight decay 5e-4),
 * 200 epochs, each followed by the loss and accuracy on the training and validation papers.
 *
 * <pre>run.sh GcnCora [epochs]</pre>
 */
public final class GcnCora {

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 200);
        Path dir = cora();
        List<String> content = Files.readAllLines(dir.resolve("cora.content"));
        Map<String, Integer> node = new HashMap<>();
        Map<String, Integer> topics = new LinkedHashMap<>();
        int n = content.size();
        int features = content.get(0).split("\t").length - 2;
        float[] x = new float[n * features];
        int[] label = new int[n];
        for (int i = 0; i < n; i++) {
            String[] f = content.get(i).split("\t");
            node.put(f[0], i);
            float sum = 0;
            for (int j = 0; j < features; j++) {
                x[i * features + j] = Float.parseFloat(f[j + 1]);
                sum += x[i * features + j];
            }
            for (int j = 0; j < features && sum > 0; j++) {
                x[i * features + j] /= sum;          // row-normalised features, as in the paper
            }
            label[i] = topics.computeIfAbsent(f[features + 1], k -> topics.size());
        }
        List<int[]> edges = new ArrayList<>();
        for (String line : Files.readAllLines(dir.resolve("cora.cites"))) {
            String[] f = line.split("\t");
            Integer a = node.get(f[0]);
            Integer b = node.get(f[1]);
            if (a != null && b != null) {
                edges.add(new int[] { a, b });     // a cited paper gathers from the papers citing it, as the original's A[cited, citing]
            }
        }
        int[] src = edges.stream().mapToInt(e -> e[1]).toArray();
        int[] dst = edges.stream().mapToInt(e -> e[0]).toArray();
        int classes = topics.size();
        System.out.printf("%,d papers, %,d features, %,d citations, %d topics%n", n, features, edges.size(), classes);

        // a random order: the first 1000 papers for the test, the next 500 for validation, the rest for training
        int[] order = Examples.permutation(n, 42);
        int[] testIdx = java.util.Arrays.copyOfRange(order, 0, 1000);
        int[] valIdx = java.util.Arrays.copyOfRange(order, 1000, 1500);
        int[] trainIdx = java.util.Arrays.copyOfRange(order, 1500, n);

        SparseMatrix.Valued adj = SparseMatrix.gcn(n, src, dst);
        Module gcn = new Sequential(new GraphConv(features, 16, 2).graph(adj), new ReLU(), new Dropout(0.5f, 3), new GraphConv(16, classes, 4).graph(adj));
        Adam opt = new Adam(gcn.parameters(), 0.01f, 0.9f, 0.999f, 1e-8f, 5e-4f, false);
        Tensor features0 = Tensor.of(x, n, features);
        Tensor trainRows = Tensor.of(trainIdx, trainIdx.length);
        Tensor trainLabels = Tensor.of(pick(label, trainIdx), trainIdx.length);

        long t0 = System.nanoTime();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            gcn.train(true);
            // the whole graph goes through the network; the loss reads the training papers' rows
            Tensor loss = F.crossEntropy(F.embedding(trainRows, gcn.forward(features0)), trainLabels);
            loss.backward();
            opt.step();
            loss.item();
            opt.zeroGrad();
            float trainAcc = accuracy(gcn, features0, trainIdx, label);    // the original evaluates both sets every epoch
            float valAcc = accuracy(gcn, features0, valIdx, label);
            if (epoch % 20 == 0) {
                System.out.printf("epoch %d: training accuracy %.1f%%, validation accuracy %.1f%%%n", epoch, 100 * trainAcc, 100 * valAcc);
            }
        }
        System.out.printf("test accuracy %.1f%% (%.1f s)%n", 100 * accuracy(gcn, features0, testIdx, label), Examples.seconds(t0));
    }

    static float accuracy(Module gcn, Tensor x, int[] idx, int[] label) {
        gcn.train(false);
        int[] predicted = Grad.inference(() -> F.embedding(Tensor.of(idx, idx.length), gcn.forward(x))).argmax(1).toIntArray();
        int correct = 0;
        for (int i = 0; i < idx.length; i++) {
            correct += predicted[i] == label[idx[i]] ? 1 : 0;
        }
        return (float) correct / idx.length;
    }

    static int[] pick(int[] values, int[] idx) {
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = values[idx[i]];
        }
        return out;
    }

    /** The Cora dataset (LINQS), downloaded and unpacked once. */
    static Path cora() throws IOException, InterruptedException {
        Path dir = Examples.dataDir().resolve("cora");
        if (!Files.exists(dir.resolve("cora").resolve("cora.content"))) {
            Path tgz = Examples.download("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz", dir.resolve("cora.tgz"));
            Examples.untar(tgz, dir);
        }
        return dir.resolve("cora");
    }
}
