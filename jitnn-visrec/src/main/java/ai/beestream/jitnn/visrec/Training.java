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
package ai.beestream.jitnn.visrec;

import java.util.Random;
import java.util.logging.Logger;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.optim.SGD;

/**
 * Trains a model as JSR 381's builders ask: SGD at the given learning rate (with the reference
 * implementation's momentum of 0.7 for image classifiers), until the mean loss of an epoch falls
 * below {@code maxError} or {@code maxEpochs} have run. Batches of {@link #BATCH} examples, where
 * the reference implementation updates after every example: one compiled plan serves every full
 * batch.
 */
final class Training {

    static final int BATCH = 32;

    private static final Logger LOG = Logger.getLogger(Training.class.getName());

    private Training() {
    }

    /**
     * @param x       inputs, {@code n} examples of {@code shape} each, back to back
     * @param classes class index per example for a softmax output, or null
     * @param targets {@code n * outputs} targets in [0, 1] for sigmoid outputs, or null
     */
    static void train(Module model, float[] x, int n, int[] shape, int[] classes, float[] targets, int outputs, float learningRate, float momentum, int maxEpochs,
            float maxError) {
        int size = 1;
        for (int d : shape) {
            size *= d;
        }
        SGD opt = new SGD(model.parameters(), learningRate, momentum, 0f, false);
        model.train(true);
        Random rnd = new Random(1);
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        int epochs = Math.max(1, maxEpochs);
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = n - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            double sum = 0;
            int batches = 0;
            for (int start = 0; start < n; start += BATCH) {
                int b = Math.min(BATCH, n - start);
                float[] xb = new float[b * size];
                for (int k = 0; k < b; k++) {
                    System.arraycopy(x, order[start + k] * size, xb, k * size, size);
                }
                int[] dims = new int[shape.length + 1];
                dims[0] = b;
                System.arraycopy(shape, 0, dims, 1, shape.length);
                Tensor logits = model.forward(Tensor.of(xb, dims));
                Tensor loss;
                if (classes != null) {
                    int[] yb = new int[b];
                    for (int k = 0; k < b; k++) {
                        yb[k] = classes[order[start + k]];
                    }
                    loss = F.crossEntropy(logits, Tensor.of(yb, b));
                } else {
                    float[] yb = new float[b * outputs];
                    for (int k = 0; k < b; k++) {
                        System.arraycopy(targets, order[start + k] * outputs, yb, k * outputs, outputs);
                    }
                    loss = F.binaryCrossEntropyWithLogits(logits, Tensor.of(yb, b, outputs));
                }
                loss.backward();
                opt.step();
                sum += loss.item();
                opt.zeroGrad();
                batches++;
            }
            double mean = sum / batches;
            LOG.fine(() -> "epoch loss " + mean);
            if (maxError > 0 && mean < maxError) {
                LOG.info("stopped after epoch " + epoch + ": loss " + mean + " < max error " + maxError);
                break;
            }
        }
        model.train(false);
    }
}
