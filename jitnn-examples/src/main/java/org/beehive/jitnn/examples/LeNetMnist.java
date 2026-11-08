package org.beehive.jitnn.examples;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.DataLoader;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.nn.Conv2d;
import org.beehive.jitnn.nn.Flatten;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.MaxPool2d;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.ReLU;
import org.beehive.jitnn.nn.Sequential;
import org.beehive.jitnn.nn.optim.SGD;

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
