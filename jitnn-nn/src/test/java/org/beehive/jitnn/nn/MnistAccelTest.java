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
 * Wave 1 gate: a 784-256-128-10 MLP on MNIST, compared with PyTorch started from the same
 * weights (jitNN's Philox init, reproduced in scripts/pytorch/jitnn_init.py) on the same batches.
 * Needs the IDX files in {@code -Djitnn.mnist=<dir>}; the PyTorch run is read from
 * {@code <dir>/mnist_mlp_pytorch.f32}, written by {@code scripts/pytorch/mnist_mlp.py}.
 */
class MnistAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void matchesPyTorch() throws Exception {
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
        java.util.List<Float> losses = new java.util.ArrayList<>();
        for (int epoch = 0; epoch < 2; epoch++) {
            for (int b = 0; b < train.count / batch; b++) {
                train.batch(b, batch, xb, yb);
                Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, batch, Mnist.PIXELS)), Tensor.of(yb, batch));
                loss.backward();
                opt.step();
                losses.add(loss.item());
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

        Path reference = Path.of(dir, "mnist_mlp_pytorch.f32");
        assumeTrue(java.nio.file.Files.exists(reference), "run scripts/pytorch/mnist_mlp.py to write " + reference);
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(java.nio.file.Files.readAllBytes(reference)).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float torchAccuracy = buf.getFloat();
        float[] torchLosses = new float[buf.remaining() / 4];
        buf.asFloatBuffer().get(torchLosses);
        assertTrue(torchLosses.length == losses.size(), "PyTorch ran " + torchLosses.length + " steps, jitNN " + losses.size());
        // Early steps agree tightly; later ones drift apart as FP32 rounding differences compound.
        double worstEarly = 0;
        for (int i = 0; i < 50; i++) {
            worstEarly = Math.max(worstEarly, Math.abs(losses.get(i) - torchLosses[i]) / Math.abs(torchLosses[i]));
        }
        double jitnnMean = 0;
        double torchMean = 0;
        for (int i = losses.size() - 100; i < losses.size(); i++) {
            jitnnMean += losses.get(i) / 100.0;
            torchMean += torchLosses[i] / 100.0;
        }
        System.out.printf("[jitnn] vs PyTorch: worst relative loss gap over steps 0-49 %.2e; mean loss of the last 100 steps %.4f vs %.4f; accuracy %.2f%% vs %.2f%%%n",
                worstEarly, jitnnMean, torchMean, 100 * accuracy, 100 * torchAccuracy);
        assertTrue(worstEarly < 1e-3, "early losses differ by " + worstEarly);
        assertTrue(Math.abs(accuracy - torchAccuracy) <= 0.003, "accuracy " + accuracy + " vs PyTorch " + torchAccuracy);
    }
}
