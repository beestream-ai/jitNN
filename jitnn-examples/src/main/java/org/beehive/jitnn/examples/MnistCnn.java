package org.beehive.jitnn.examples;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.DataLoader;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.nn.Conv2d;
import org.beehive.jitnn.nn.Dropout;
import org.beehive.jitnn.nn.Flatten;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.MaxPool2d;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.ReLU;
import org.beehive.jitnn.nn.Sequential;
import org.beehive.jitnn.nn.optim.AdaDelta;
import org.beehive.jitnn.nn.optim.LRScheduler;

/**
 * Handwritten digits, PyTorch's MNIST example with its settings: two 3x3 convolutions (32 and 64
 * channels), max pooling, dropout 0.25 and 0.5, a 128-unit dense layer, AdaDelta (lr 1.0) with
 * the learning rate times 0.7 after every epoch, batches of 64, 14 epochs, the test set in
 * batches of 1000 after every epoch.
 *
 * <pre>run.sh MnistCnn [epochs]</pre>
 */
public final class MnistCnn {

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 14);
        int batch = 64;
        Mnist train = Mnist.load(Examples.mnist(), true);
        Mnist test = Mnist.load(Examples.mnist(), false);

        Module model = new Sequential(
                new Conv2d(1, 32, 3, 1, 0, 1), new ReLU(), new Conv2d(32, 64, 3, 1, 0, 2), new ReLU(), new MaxPool2d(2), new Dropout(0.25f, 3),
                new Flatten(), new Linear(9216, 128, 4), new ReLU(), new Dropout(0.5f, 5), new Linear(128, 10, 6));
        AdaDelta opt = new AdaDelta(model.parameters(), 1.0f, 0.9f, 1e-6f, 0);
        LRScheduler schedule = LRScheduler.step(opt, 1, 0.7f);

        try (DataLoader loader = DataLoader.builder(train, batch).shuffle(1).build()) {
            for (int epoch = 1; epoch <= epochs; epoch++) {
                long t0 = System.nanoTime();
                model.train(true);
                float sum = 0;
                for (DataLoader.Batch b : loader.epoch(epoch)) {
                    // log_softmax + nll_loss, as the original, is cross-entropy
                    Tensor loss = F.crossEntropy(model.forward(b.x()), b.y());
                    loss.backward();
                    opt.step();
                    sum += loss.item();
                    opt.zeroGrad();
                }
                float acc = accuracy(model, test);
                System.out.printf("epoch %d: loss %.4f, test accuracy %.2f%%, %.2f s%n", epoch, sum / loader.batches(), 100 * acc, Examples.seconds(t0));
                schedule.step();
            }
        }
    }

    static float accuracy(Module model, Mnist test) {
        model.train(false);
        int n = 1000;
        float correct = 0;
        try (DataLoader loader = DataLoader.builder(test, n).build()) {
            for (DataLoader.Batch b : loader.epoch(0)) {
                correct += F.accuracy(Grad.inference(() -> model.forward(b.x())), b.y()) * n;
            }
        }
        return correct / test.count();
    }
}
