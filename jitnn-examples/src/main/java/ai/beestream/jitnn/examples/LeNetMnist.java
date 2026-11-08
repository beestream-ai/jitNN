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
import ai.beestream.jitnn.nn.Conv2d;
import ai.beestream.jitnn.nn.Flatten;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.MaxPool2d;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.ReLU;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.optim.SGD;

/**
 * Handwritten digits with a convolutional network, after DL4J's LeNetMNIST: two 5x5 convolutions
 * (20 and 50 channels) with max pooling, a 500-unit dense layer, SGD with Nesterov momentum.
 *
 * <pre>run.sh LeNetMnist [epochs]</pre>
 */
public final class LeNetMnist {

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 3);
        int batch = 64;
        Mnist train = Mnist.load(Examples.mnist(), true);
        Mnist test = Mnist.load(Examples.mnist(), false);

        Module model = new Sequential(
                new Conv2d(1, 20, 5, 1, 0, 1), new MaxPool2d(2),        // [B, 20, 12, 12]
                new Conv2d(20, 50, 5, 1, 0, 2), new MaxPool2d(2),       // [B, 50, 4, 4]
                new Flatten(), new Linear(800, 500, 3), new ReLU(), new Linear(500, 10, 4));
        SGD opt = new SGD(model.parameters(), 0.01f, 0.9f, 5e-4f, true);
        System.out.printf("LeNet with %,d parameters%n", model.parameterCount());

        try (DataLoader loader = DataLoader.builder(train, batch).shuffle(42).dropLast(true).build()) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                long t0 = System.nanoTime();
                model.train(true);
                float sum = 0;
                for (DataLoader.Batch b : loader.epoch(epoch)) {
                    Tensor loss = F.crossEntropy(model.forward(b.x().reshape(batch, 1, 28, 28)), b.y());
                    loss.backward();
                    opt.step();
                    sum += loss.item();
                    opt.zeroGrad();
                }
                System.out.printf("epoch %d: loss %.4f, test accuracy %.2f%%, %.1f s%n", epoch + 1, sum / loader.batches(), 100 * accuracy(model, test), Examples.seconds(t0));
            }
        }
    }

    static float accuracy(Module model, Mnist test) {
        model.train(false);
        int n = 1000;
        float correct = 0;
        try (DataLoader loader = DataLoader.builder(test, n).build()) {
            for (DataLoader.Batch b : loader.epoch(0)) {
                Tensor logits = Grad.inference(() -> model.forward(b.x().reshape(n, 1, 28, 28)));
                correct += F.accuracy(logits, b.y()) * n;
            }
        }
        return correct / test.count();
    }
}
