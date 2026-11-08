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
import ai.beestream.jitnn.data.Cifar10;
import ai.beestream.jitnn.data.DataLoader;
import ai.beestream.jitnn.data.Transforms;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.models.ResNet;
import ai.beestream.jitnn.nn.optim.SGD;

/**
 * CIFAR-10 with ResNet-18, after DJL's TrainResnetWithCifar10: random crops and flips, SGD with
 * momentum, batches of 128 assembled by four threads while the device trains. The dataset (170
 * MB) is downloaded on first use.
 *
 * <pre>run.sh Cifar10ResNet [epochs]</pre>
 */
public final class Cifar10ResNet {

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 2);
        int batch = 128;
        Cifar10 train = Cifar10.load(Examples.cifar10(), true);
        Cifar10 test = Cifar10.load(Examples.cifar10(), false);

        Module model = ResNet.resnet18(3, 10, true, 42);   // small: a 3x3 stem for 32x32 images
        SGD opt = new SGD(model.parameters(), 0.05f, 0.9f, 5e-4f, false);
        System.out.printf("ResNet-18 with %,d parameters, %d training images%n", model.parameterCount(), train.count());

        Transforms.Transform augment = Transforms.randomCrop(4).then(Transforms.randomHorizontalFlip());
        try (DataLoader loader = DataLoader.builder(train, batch).shuffle(42).dropLast(true).transform(augment).workers(4).build()) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                long t0 = System.nanoTime();
                model.train(true);
                float sum = 0;
                int steps = 0;
                for (DataLoader.Batch b : loader.epoch(epoch)) {
                    Tensor loss = F.crossEntropy(model.forward(b.x()), b.y());
                    loss.backward();
                    opt.step();
                    sum += loss.item();
                    opt.zeroGrad();
                    if (++steps % 100 == 0) {
                        System.out.printf("  step %d: loss %.4f, %.0f images/s%n", steps, sum / steps, steps * batch / Examples.seconds(t0));
                    }
                }
                System.out.printf("epoch %d: loss %.4f, test accuracy %.2f%%, %.1f s%n", epoch + 1, sum / steps, 100 * accuracy(model, test), Examples.seconds(t0));
            }
        }
    }

    static float accuracy(Module model, Cifar10 test) {
        model.train(false);
        int n = 250;
        float correct = 0;
        try (DataLoader loader = DataLoader.builder(test, n).build()) {
            for (DataLoader.Batch b : loader.epoch(0)) {
                Tensor logits = Grad.inference(() -> model.forward(b.x()));
                correct += F.accuracy(logits, b.y()) * n;
            }
        }
        return correct / test.count();
    }
}
