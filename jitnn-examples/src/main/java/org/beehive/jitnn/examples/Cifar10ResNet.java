package org.beehive.jitnn.examples;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Cifar10;
import org.beehive.jitnn.data.DataLoader;
import org.beehive.jitnn.data.Transforms;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.models.ResNet;
import org.beehive.jitnn.nn.optim.SGD;

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
