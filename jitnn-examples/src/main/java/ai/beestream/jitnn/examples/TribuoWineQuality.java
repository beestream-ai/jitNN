package ai.beestream.jitnn.examples;

import java.util.Arrays;
import java.util.List;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.ml.GradientBoosting;
import ai.beestream.jitnn.ml.LinearRegression;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.ReLU;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.optim.Adam;

/**
 * Predicts the quality of red wines from their chemistry, after Tribuo's regression tutorial:
 * 1,599 wines, 11 measurements, quality scored 3 to 8. Ridge regression (closed form),
 * gradient-boosted trees and a small neural network, compared by RMSE, MAE and R^2 on a held-out
 * 30%. Data: P. Cortez et al., UCI Machine Learning Repository (CC BY 4.0).
 *
 * <pre>run.sh TribuoWineQuality</pre>
 */
public final class TribuoWineQuality {

    public static void main(String[] args) throws Exception {
        List<String[]> rows = Examples.csv(Examples.download("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                Examples.dataDir().resolve("wine").resolve("winequality-red.csv")), ";");
        int n = rows.size();
        int d = 11;
        int[] order = Examples.permutation(n, 1);
        float[] x = new float[n * d];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            String[] f = rows.get(order[i]);
            for (int j = 0; j < d; j++) {
                x[i * d + j] = Float.parseFloat(f[j]);
            }
            y[i] = Float.parseFloat(f[d]);
        }
        int trainN = n * 7 / 10;
        int testN = n - trainN;
        Examples.standardise(x, n, d, trainN);
        Tensor trainX = Tensor.of(Arrays.copyOfRange(x, 0, trainN * d), trainN, d);
        Tensor trainY = Tensor.of(Arrays.copyOfRange(y, 0, trainN), trainN);
        Tensor testX = Tensor.of(Arrays.copyOfRange(x, trainN * d, n * d), testN, d);
        float[] testY = Arrays.copyOfRange(y, trainN, n);
        System.out.printf("%,d wines: %,d for training, %d held out%n%n", n, trainN, testN);

        long t0 = System.nanoTime();
        report("ridge regression", new LinearRegression(1.0f, true).fit(trainX, trainY).predict(testX).toFloatArray(), testY, t0);

        t0 = System.nanoTime();
        report("gradient boosting (300 trees)", new GradientBoosting(false, 300, 0.05f, 5).fit(trainX, trainY).predictValues(testX).toFloatArray(), testY, t0);

        // the network: the last 20% of the training wines decide when to stop (early stopping, as Keras' EarlyStopping)
        t0 = System.nanoTime();
        int fitN = trainN * 4 / 5;
        Tensor fitX = trainX.narrow(0, 0, fitN);
        Tensor fitY = trainY.narrow(0, 0, fitN).reshape(fitN, 1);
        Tensor valX = trainX.narrow(0, fitN, trainN - fitN);
        Tensor valY = trainY.narrow(0, fitN, trainN - fitN).reshape(trainN - fitN, 1);
        Module net = new Sequential(new Linear(d, 64, 1), new ReLU(), new Linear(64, 32, 2), new ReLU(), new Linear(32, 1, 3));
        Adam opt = Adam.adamW(net.parameters(), 3e-3f, 1e-2f);
        float best = Float.MAX_VALUE;
        ai.beestream.jitnn.io.StateDict bestWeights = null;
        int bestStep = 0;
        for (int step = 1; step <= 2000 && step - bestStep <= 200; step++) {
            net.train(true);
            Tensor loss = F.mseLoss(net.forward(fitX), fitY);
            loss.backward();
            opt.step();
            loss.item();
            opt.zeroGrad();
            if (step % 10 == 0) {
                net.train(false);
                float val = F.mseLoss(Grad.inference(() -> net.forward(valX)), valY).item();
                if (val < best) {
                    best = val;
                    bestStep = step;
                    bestWeights = snapshot(net);
                }
            }
        }
        net.loadStateDict(bestWeights);
        net.train(false);
        report("neural network (stopped at step " + bestStep + ")", Grad.inference(() -> net.forward(testX)).toFloatArray(), testY, t0);
    }

    /** A host copy of the network's weights (a state dict holds the live parameters, which keep training). */
    static ai.beestream.jitnn.io.StateDict snapshot(Module net) {
        ai.beestream.jitnn.io.StateDict d = new ai.beestream.jitnn.io.StateDict();
        List<Tensor> ps = net.parameters();
        for (int i = 0; i < ps.size(); i++) {
            d.put("parameters." + i, Tensor.of(ps.get(i).toFloatArray(), ps.get(i).shape().dims()));
        }
        return d;
    }

    static void report(String name, float[] predicted, float[] truth, long t0) {
        double mean = 0;
        for (float t : truth) {
            mean += t / truth.length;
        }
        double se = 0;
        double ae = 0;
        double total = 0;
        for (int i = 0; i < truth.length; i++) {
            se += (predicted[i] - truth[i]) * (predicted[i] - truth[i]);
            ae += Math.abs(predicted[i] - truth[i]);
            total += (truth[i] - mean) * (truth[i] - mean);
        }
        System.out.printf("%-30s RMSE %.3f  MAE %.3f  R^2 %.3f  (%.2f s)%n", name, Math.sqrt(se / truth.length), ae / truth.length, 1 - se / total, Examples.seconds(t0));
    }
}
