package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Wave 1 gate: a 784-256-128-10 MLP on MNIST. Needs the IDX files in the directory given by
 * {@code -Djitnn.mnist=<dir>}; skipped otherwise.
 */
class MnistAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void mlpReaches97Percent() throws Exception {
        String dir = System.getProperty("jitnn.mnist", "");
        assumeTrue(!dir.isEmpty() && Mnist.isPresent(Path.of(dir)), "set -Djitnn.mnist=<dir with the MNIST IDX files>");
        Mnist train = Mnist.load(Path.of(dir), true);
        Mnist test = Mnist.load(Path.of(dir), false);

        int batch = 128;
        Sequential model = new Sequential(new Linear(784, 256, 1), new ReLU(), new Linear(256, 128, 2), new ReLU(), new Linear(128, 10, 3));
        Adam opt = new Adam(model.parameters(), 1e-3f);
        float[] xb = new float[batch * Mnist.PIXELS];
        int[] yb = new int[batch];
        long t0 = System.nanoTime();
        int steps = 0;
        for (int epoch = 0; epoch < 2; epoch++) {
            for (int b = 0; b < train.count / batch; b++) {
                train.batch(b, batch, xb, yb);
                Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, batch, Mnist.PIXELS)), Tensor.of(yb, batch));
                loss.backward();
                opt.step();
                loss.item();
                opt.zeroGrad();
                steps++;
            }
        }
        double seconds = (System.nanoTime() - t0) / 1e9;

        int evalBatch = 1000;
        float[] xe = new float[evalBatch * Mnist.PIXELS];
        int[] ye = new int[evalBatch];
        double correct = 0;
        for (int b = 0; b < test.count / evalBatch; b++) {
            test.batch(b, evalBatch, xe, ye);
            correct += F.accuracy(model.forward(Tensor.of(xe, evalBatch, Mnist.PIXELS)), Tensor.of(ye, evalBatch)) * evalBatch;
        }
        double accuracy = correct / test.count;
        System.out.printf("[jitnn] MNIST MLP: %d steps in %.1f s (%.0f samples/s), test accuracy %.2f%%%n", steps, seconds, steps * batch / seconds, 100 * accuracy);
        assertTrue(accuracy > 0.97, "test accuracy " + accuracy);
    }
}
