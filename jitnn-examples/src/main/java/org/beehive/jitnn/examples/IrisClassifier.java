package org.beehive.jitnn.examples;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ml.LogisticRegression;
import org.beehive.jitnn.ml.RandomForest;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.Sequential;
import org.beehive.jitnn.nn.Tanh;
import org.beehive.jitnn.nn.optim.Adam;

/**
 * Iris flowers, after DL4J's IrisClassifier and Deep Netts' iris example: 150 flowers, 4
 * measurements, 3 species. A small network (4-16-3, tanh) trains on 80% of the flowers in full
 * batches, then a random forest and logistic regression from {@code org.beehive.jitnn.ml} on the
 * same split. Data: R. A. Fisher's iris data from the UCI Machine Learning Repository (CC BY 4.0).
 *
 * <pre>run.sh IrisClassifier</pre>
 */
public final class IrisClassifier {

    private static final String[] SPECIES = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

    public static void main(String[] args) throws Exception {
        List<float[]> rows = new ArrayList<>();
        List<Integer> species = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(Examples.resource("iris.data"), StandardCharsets.US_ASCII))) {
            for (String line; (line = r.readLine()) != null;) {
                String[] f = line.split(",");
                if (f.length == 5) {
                    rows.add(new float[] { Float.parseFloat(f[0]), Float.parseFloat(f[1]), Float.parseFloat(f[2]), Float.parseFloat(f[3]) });
                    species.add(List.of(SPECIES).indexOf(f[4]));
                }
            }
        }
        // shuffle, standardise each measurement, split 120 / 30
        int n = rows.size();
        int[] order = new Random(7).ints(0, Integer.MAX_VALUE).limit(n).toArray();
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (a, b) -> Integer.compare(order[a], order[b]));
        float[] mean = new float[4];
        float[] std = new float[4];
        for (float[] row : rows) {
            for (int j = 0; j < 4; j++) {
                mean[j] += row[j] / n;
            }
        }
        for (float[] row : rows) {
            for (int j = 0; j < 4; j++) {
                std[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
            }
        }
        int trainN = 120;
        int testN = n - trainN;
        float[] x = new float[n * 4];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            float[] row = rows.get(idx[i]);
            for (int j = 0; j < 4; j++) {
                x[i * 4 + j] = (row[j] - mean[j]) / (float) Math.sqrt(std[j]);
            }
            y[i] = species.get(idx[i]);
        }
        Tensor trainX = Tensor.of(java.util.Arrays.copyOfRange(x, 0, trainN * 4), trainN, 4);
        Tensor trainY = Tensor.of(java.util.Arrays.copyOfRange(y, 0, trainN), trainN);
        Tensor testX = Tensor.of(java.util.Arrays.copyOfRange(x, trainN * 4, n * 4), testN, 4);
        int[] testY = java.util.Arrays.copyOfRange(y, trainN, n);

        Module net = new Sequential(new Linear(4, 16, 1), new Tanh(), new Linear(16, 3, 2));
        Adam opt = new Adam(net.parameters(), 0.01f);
        for (int step = 1; step <= 300; step++) {
            Tensor loss = F.crossEntropy(net.forward(trainX), trainY);
            loss.backward();
            opt.step();
            float value = loss.item();
            opt.zeroGrad();
            if (step % 100 == 0) {
                System.out.printf("step %d: loss %.4f%n", step, value);
            }
        }
        net.train(false);
        int[] netPredicted = Grad.inference(() -> net.forward(testX)).argmax(1).toIntArray();
        report("neural network (4-16-3)", netPredicted, testY);

        int[] trainLabels = java.util.Arrays.copyOfRange(y, 0, trainN);
        report("random forest (100 trees)", new RandomForest(RandomForest.Task.CLASSIFICATION, 100, 8, 7).fit(trainX, trainLabels).predict(testX), testY);
        report("logistic regression", new LogisticRegression().fit(trainX, trainLabels).predict(testX), testY);
    }

    private static void report(String name, int[] predicted, int[] truth) {
        int correct = 0;
        int[][] confusion = new int[3][3];
        for (int i = 0; i < truth.length; i++) {
            correct += predicted[i] == truth[i] ? 1 : 0;
            confusion[truth[i]][predicted[i]]++;
        }
        System.out.printf("%n%s: %d of %d test flowers right (%.1f%%)%n", name, correct, truth.length, 100.0 * correct / truth.length);
        System.out.println("  confusion (rows: true species, columns: predicted)");
        for (int s = 0; s < 3; s++) {
            System.out.printf("  %-16s %3d %3d %3d%n", SPECIES[s], confusion[s][0], confusion[s][1], confusion[s][2]);
        }
    }
}
