package ai.beestream.jitnn.examples;

import java.util.Arrays;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ml.GradientBoosting;
import ai.beestream.jitnn.ml.LogisticRegression;
import ai.beestream.jitnn.ml.RandomForest;

/**
 * Predicts whether a prostate tumour has penetrated the capsule, after H2O's prostate GLM, GBM and
 * random-forest walkthroughs: 380 patients, 7 measurements (age, race, rectal examination, PSA,
 * tumour volume, Gleason score). Logistic regression (H2O's binomial GLM), gradient-boosted trees
 * and a random forest from {@code ai.beestream.jitnn.ml}, each scored by accuracy and AUC on a
 * held-out 20%.
 *
 * <pre>run.sh H2oProstate</pre>
 */
public final class H2oProstate {

    public static void main(String[] args) throws Exception {
        List<String[]> rows = Examples.csv(Examples.download("https://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv",
                Examples.dataDir().resolve("h2o").resolve("prostate.csv")), ",");
        // ID, CAPSULE, AGE, RACE, DPROS, DCAPS, PSA, VOL, GLEASON
        int n = rows.size();
        int d = 7;
        int[] order = Examples.permutation(n, 1234);
        float[] x = new float[n * d];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            String[] f = rows.get(order[i]);
            y[i] = Integer.parseInt(f[1]);
            for (int j = 0; j < d; j++) {
                x[i * d + j] = Float.parseFloat(f[j + 2]);
            }
        }
        int trainN = n * 4 / 5;
        Examples.standardise(x, n, d, trainN);
        Tensor trainX = Tensor.of(Arrays.copyOfRange(x, 0, trainN * d), trainN, d);
        Tensor testX = Tensor.of(Arrays.copyOfRange(x, trainN * d, n * d), n - trainN, d);
        int[] trainY = Arrays.copyOfRange(y, 0, trainN);
        int[] testY = Arrays.copyOfRange(y, trainN, n);
        System.out.printf("%d patients: %d for training, %d held out%n%n", n, trainN, n - trainN);

        long t0 = System.nanoTime();
        LogisticRegression glm = new LogisticRegression().fit(trainX, trainY);
        report("logistic regression (GLM)", glm.predictProba(testX), testY, t0);

        t0 = System.nanoTime();
        GradientBoosting gbm = new GradientBoosting(true, 100, 0.05f, 3).fit(trainX, trainY);
        report("gradient boosting (GBM, 100 trees)", gbm.predictProba(testX), testY, t0);

        t0 = System.nanoTime();
        RandomForest drf = new RandomForest(RandomForest.Task.CLASSIFICATION, 200, 8, 7).fit(trainX, trainY);
        int[] predicted = drf.predict(testX);
        float[] votes = new float[predicted.length * 2];
        for (int i = 0; i < predicted.length; i++) {
            votes[i * 2 + predicted[i]] = 1;
        }
        report("random forest (DRF, 200 trees)", Tensor.of(votes, predicted.length, 2), testY, t0);
    }

    /** Accuracy at 0.5 and the area under the ROC curve of the probabilities of class 1. */
    static void report(String name, Tensor proba, int[] truth, long t0) {
        float[] p = proba.toFloatArray();
        int m = truth.length;
        float[] score = new float[m];
        int correct = 0;
        for (int i = 0; i < m; i++) {
            score[i] = p[i * 2 + 1];
            correct += (score[i] >= 0.5f ? 1 : 0) == truth[i] ? 1 : 0;
        }
        System.out.printf("%-36s accuracy %.1f%%  AUC %.3f  (%.2f s)%n", name, 100.0 * correct / m, auc(score, truth), Examples.seconds(t0));
    }

    /** AUC as the probability that a positive scores above a negative (ties count half). */
    static double auc(float[] score, int[] truth) {
        double wins = 0;
        long pairs = 0;
        for (int i = 0; i < score.length; i++) {
            for (int j = 0; j < score.length; j++) {
                if (truth[i] == 1 && truth[j] == 0) {
                    pairs++;
                    wins += score[i] > score[j] ? 1 : score[i] == score[j] ? 0.5 : 0;
                }
            }
        }
        return wins / pairs;
    }
}
