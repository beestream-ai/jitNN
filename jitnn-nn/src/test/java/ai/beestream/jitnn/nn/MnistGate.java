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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.Mnist;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.optim.Adam;

/**
 * Trains a model on MNIST and compares it with a PyTorch run from the same initial weights on the
 * same batches (scripts/pytorch/mnist_train.py): the early losses step by step, and the test
 * accuracy. Needs {@code -Djitnn.mnist=<dir>} with the IDX files and the PyTorch result file.
 */
final class MnistGate {

    private MnistGate() {
    }

    static void run(String name, Module model, int[] inputShape, int epochs, String referenceFile, double accuracySlack) throws Exception {
        String dir = System.getProperty("jitnn.mnist", "");
        assumeTrue(!dir.isEmpty() && Mnist.isPresent(Path.of(dir)), "set -Djitnn.mnist=<dir with the MNIST IDX files>");
        Path reference = Path.of(dir, referenceFile);
        assumeTrue(Files.exists(reference), "run scripts/pytorch/mnist_train.py to write " + reference);
        Mnist train = Mnist.load(Path.of(dir), true);
        Mnist test = Mnist.load(Path.of(dir), false);

        int batch = 128;
        Adam opt = new Adam(model.parameters(), 1e-3f);
        float[] xb = new float[batch * Mnist.PIXELS];
        int[] yb = new int[batch];
        int[] shape = shape(batch, inputShape);
        List<Float> losses = new ArrayList<>();
        long[] split0 = Engine.timeSplit();
        long t0 = System.nanoTime();
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0; b < train.count / batch; b++) {
                train.batch(b, batch, xb, yb);
                Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, shape)), Tensor.of(yb, batch));
                loss.backward();
                opt.step();
                losses.add(loss.item());
                opt.zeroGrad();
            }
        }
        double seconds = (System.nanoTime() - t0) / 1e9;
        long[] split1 = Engine.timeSplit();
        int steps = losses.size();
        System.out.printf("[jitnn] %s per step: %.3f ms total, %.3f ms jitNN executor, %.3f ms TornadoVM execute%n", name, seconds * 1e3 / steps,
                (split1[0] - split0[0]) / 1e6 / steps, (split1[1] - split0[1]) / 1e6 / steps);

        model.train(false);
        int evalBatch = 1000;
        float[] xe = new float[evalBatch * Mnist.PIXELS];
        int[] ye = new int[evalBatch];
        double correct = 0;
        for (int b = 0; b < test.count / evalBatch; b++) {
            test.batch(b, evalBatch, xe, ye);
            correct += F.accuracy(model.forward(Tensor.of(xe, shape(evalBatch, inputShape))), Tensor.of(ye, evalBatch)) * evalBatch;
        }
        double accuracy = correct / test.count;

        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(reference)).order(ByteOrder.LITTLE_ENDIAN);
        float torchAccuracy = buf.getFloat();
        float[] torchLosses = new float[buf.remaining() / 4];
        buf.asFloatBuffer().get(torchLosses);
        assertTrue(torchLosses.length == steps, "PyTorch ran " + torchLosses.length + " steps, jitNN " + steps);
        double worstEarly = 0;
        int worstAt = 0;
        for (int i = 0; i < 50; i++) {
            double gap = Math.abs(losses.get(i) - torchLosses[i]) / Math.abs(torchLosses[i]);
            if (gap > worstEarly) {
                worstEarly = gap;
                worstAt = i;
            }
        }
        System.out.printf("[jitnn] %s first losses %s vs PyTorch %.6f %.6f %.6f; worst gap at step %d%n", name, losses.subList(0, 3), torchLosses[0], torchLosses[1], torchLosses[2], worstAt);
        double jitnnMean = 0;
        double torchMean = 0;
        for (int i = steps - 100; i < steps; i++) {
            jitnnMean += losses.get(i) / 100.0;
            torchMean += torchLosses[i] / 100.0;
        }
        System.out.printf("[jitnn] %s: %d steps in %.1f s (%.0f samples/s); vs PyTorch: worst relative loss gap over steps 0-49 %.2e, mean loss of the last 100 steps %.4f vs %.4f, accuracy %.2f%% vs %.2f%%%n",
                name, steps, seconds, steps * batch / seconds, worstEarly, jitnnMean, torchMean, 100 * accuracy, 100 * torchAccuracy);
        // Early steps catch initialisation and semantic differences; later ones drift as FP32
        // rounding compounds (faster with max pooling, where tiny differences flip a window's argmax).
        double worstFirst = 0;
        for (int i = 0; i < 10; i++) {
            worstFirst = Math.max(worstFirst, Math.abs(losses.get(i) - torchLosses[i]) / Math.abs(torchLosses[i]));
        }
        assertTrue(worstFirst < 1e-4, name + ": losses of steps 0-9 differ by " + worstFirst);
        assertTrue(worstEarly < 1e-2, name + ": losses of steps 0-49 differ by " + worstEarly);
        assertTrue(Math.abs(jitnnMean - torchMean) <= 0.05 * torchMean, name + ": final mean loss " + jitnnMean + " vs " + torchMean);
        assertTrue(Math.abs(accuracy - torchAccuracy) <= accuracySlack, name + ": accuracy " + accuracy + " vs PyTorch " + torchAccuracy);
    }

    private static int[] shape(int batch, int[] inputShape) {
        int[] s = new int[inputShape.length + 1];
        s[0] = batch;
        System.arraycopy(inputShape, 0, s, 1, inputShape.length);
        return s;
    }
}
