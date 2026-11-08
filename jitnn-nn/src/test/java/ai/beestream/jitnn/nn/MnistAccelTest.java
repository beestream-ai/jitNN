package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * The MNIST gates, against PyTorch from the same weights: the wave 1 MLP (2 epochs) and the wave 2
 * LeNet-5 (1 epoch). Needs {@code -Djitnn.mnist=<dir>}; see scripts/pytorch/mnist_train.py.
 */
class MnistAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void mlpMatchesPyTorch() throws Exception {
        Sequential mlp = new Sequential(new Linear(784, 256, 1), new ReLU(), new Linear(256, 128, 2), new ReLU(), new Linear(128, 10, 3));
        MnistGate.run("MLP", mlp, new int[] { 784 }, 2, "mnist_mlp_pytorch.f32", 0.003);
    }

    @Test
    void leNetMatchesPyTorch() throws Exception {
        Sequential lenet = new Sequential(
                new Conv2d(1, 6, 5, 1, 2, 1), new ReLU(), new MaxPool2d(2),
                new Conv2d(6, 16, 5, 1, 0, 2), new ReLU(), new MaxPool2d(2),
                new Flatten(),
                new Linear(400, 120, 3), new ReLU(), new Linear(120, 84, 4), new ReLU(), new Linear(84, 10, 5));
        MnistGate.run("LeNet-5", lenet, new int[] { 1, 28, 28 }, 1, "lenet_pytorch.f32", 0.003);
    }
}
