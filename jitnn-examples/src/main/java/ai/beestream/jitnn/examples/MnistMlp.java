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
package ai.beestream.jitnn.examples;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.DataLoader;
import ai.beestream.jitnn.data.Mnist;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.ReLU;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.optim.Adam;

/**
 * Handwritten digits with a multilayer perceptron, after DL4J's MLPMnistTwoLayerExample and Deep
 * Netts' MNIST classifier: 784-500-100-10 with ReLU, Adam, cross-entropy.
 *
 * <pre>run.sh MnistMlp [epochs]</pre>
 */
public final class MnistMlp {

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 3);
        int batch = 64;
        Mnist train = Mnist.load(Examples.mnist(), true);
        Mnist test = Mnist.load(Examples.mnist(), false);

        Module model = new Sequential(new Linear(784, 500, 1), new ReLU(), new Linear(500, 100, 2), new ReLU(), new Linear(100, 10, 3));
        Adam opt = new Adam(model.parameters(), 1e-3f);
        System.out.printf("MLP with %,d parameters, %d training images%n", model.parameterCount(), train.count());

        try (DataLoader loader = DataLoader.builder(train, batch).shuffle(42).dropLast(true).build()) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                long t0 = System.nanoTime();
                model.train(true);
                float sum = 0;
                for (DataLoader.Batch b : loader.epoch(epoch)) {
                    // the first step compiles forward, backward and Adam into one plan; later steps replay it
                    Tensor loss = F.crossEntropy(model.forward(b.x().reshape(batch, 784)), b.y());
                    loss.backward();
                    opt.step();
                    sum += loss.item();
                    opt.zeroGrad();
                }
                System.out.printf("epoch %d: loss %.4f, test accuracy %.2f%%, %.1f s%n", epoch + 1, sum / loader.batches(), 100 * accuracy(model, test), Examples.seconds(t0));
            }
        }
    }

    /** Test accuracy, in batches of 1000 without gradients. */
    static float accuracy(Module model, Mnist test) {
        model.train(false);
        int n = 1000;
        float correct = 0;
        try (DataLoader loader = DataLoader.builder(test, n).build()) {
            for (DataLoader.Batch b : loader.epoch(0)) {
                Tensor logits = Grad.inference(() -> model.forward(b.x().reshape(n, 784)));
                correct += F.accuracy(logits, b.y()) * n;
            }
        }
        return correct / test.count();
    }
}
