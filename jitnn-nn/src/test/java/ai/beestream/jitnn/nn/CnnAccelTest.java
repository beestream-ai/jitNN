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
import ai.beestream.jitnn.data.Dataset;
import ai.beestream.jitnn.data.Mnist;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.models.MobileNetV2;
import ai.beestream.jitnn.nn.models.ResNet;
import ai.beestream.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * CNN gates: ResNet-18 and MobileNetV2 (small variants) for 200 MNIST steps against PyTorch
 * from the same initial weights (scripts/pytorch/cnn_train.py): losses step by step, then
 * accuracy in evaluation mode (running BatchNorm statistics) on the first 1000 test images.
 * The step-0 gradients are held to PyTorch's own FP32 error: with vendor libraries (-Djitnn.native),
 * cuDNN convolves in TF32 forward and backward on Ampere and newer, as PyTorch does by default, so
 * the gates need NVIDIA_TF32_OVERRIDE=0 there and are skipped otherwise.
 */
class CnnAccelTest {

    private static final int STEPS = 200;
    private static final int BATCH = 64;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @org.junit.jupiter.api.AfterEach
    void releasePlans() {
        if (!Boolean.getBoolean("jitnn.test.keepPlans")) {
            Engine.reset();
        }
    }

    @Test
    void resNet18MatchesPyTorch() throws Exception {
        runMnist("ResNet-18", ResNet.resnet18(1, 10, true, 1), "resnet18_pytorch.f32");
    }

    @Test
    void mobileNetV2MatchesPyTorch() throws Exception {
        runMnist("MobileNetV2", MobileNetV2.mobilenetV2(1, 10, true, 0, 1), "mobilenetv2_pytorch.f32");
    }

    /** CIFAR-10, one epoch (781 steps of 64), evaluation on all 10,000 test images. Needs -Djitnn.cifar10=<dir>. */
    @Test
    void resNet18Cifar10MatchesPyTorch() throws Exception {
        runCifar("ResNet-18 CIFAR-10", ResNet.resnet18(3, 10, true, 1), "resnet18_cifar10_pytorch.f32");
    }

    @Test
    void mobileNetV2Cifar10MatchesPyTorch() throws Exception {
        runCifar("MobileNetV2 CIFAR-10", MobileNetV2.mobilenetV2(3, 10, true, 0, 1), "mobilenetv2_cifar10_pytorch.f32");
    }

    private static void runMnist(String name, Module model, String referenceFile) throws Exception {
        String dir = System.getProperty("jitnn.mnist", "");
        assumeTrue(!dir.isEmpty() && Mnist.isPresent(Path.of(dir)), "set -Djitnn.mnist=<dir>");
        run(name, model, Mnist.load(Path.of(dir), true), Mnist.load(Path.of(dir), false), STEPS, 1000, Path.of(dir, referenceFile));
    }

    private static void runCifar(String name, Module model, String referenceFile) throws Exception {
        String dir = System.getProperty("jitnn.cifar10", "");
        assumeTrue(!dir.isEmpty() && ai.beestream.jitnn.data.Cifar10.isPresent(Path.of(dir)), "set -Djitnn.cifar10=<dir>");
        Dataset train = ai.beestream.jitnn.data.Cifar10.load(Path.of(dir), true);
        run(name, model, train, ai.beestream.jitnn.data.Cifar10.load(Path.of(dir), false), train.count() / BATCH, 10_000, Path.of(dir, referenceFile));
    }

    private static void run(String name, Module model, Dataset train, Dataset test, int steps, int evalCount, Path reference) throws Exception {
        assumeTrue(Files.exists(reference), "run scripts/pytorch/cnn_train.py to write " + reference);
        assumeTrue(!ai.beestream.jitnn.engine.Native.requested() || "0".equals(System.getenv("NVIDIA_TF32_OVERRIDE")),
                "with -Djitnn.native cuDNN convolves in TF32: set NVIDIA_TF32_OVERRIDE=0 for the FP32 gates");
        int[] shape = train.sampleShape();
        int sample = train.sampleSize();
        Adam opt = new Adam(model.parameters(), 1e-3f);
        float[] xb = new float[BATCH * sample];
        int[] yb = new int[BATCH];
        List<Float> losses = new ArrayList<>();
        long[] plans0 = Engine.planStats();
        long[] split0 = Engine.timeSplit();
        long t0 = System.nanoTime();
        long firstStep = 0;
        for (int b = 0; b < steps; b++) {
            train.batch(b, BATCH, xb, yb);
            Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, BATCH, shape[0], shape[1], shape[2])), Tensor.of(yb, BATCH));
            loss.backward();
            if (b == 0) {
                Gates.checkGradients(name, model.parameters(), reference.toString());
            }
            opt.step();
            losses.add(loss.item());
            opt.zeroGrad();
            if (b == 0) {
                firstStep = System.nanoTime() - t0;
            }
        }
        double steady = (System.nanoTime() - t0 - firstStep) / 1e6 / (steps - 1);
        long[] plans1 = Engine.planStats();
        long[] split1 = Engine.timeSplit();
        System.out.printf("[jitnn] %s per step: jitNN executor %.1f ms, TornadoVM execute %.1f ms%n", name, (split1[0] - split0[0]) / 1e6 / steps, (split1[1] - split0[1]) / 1e6 / steps);

        model.train(false);
        int chunk = 250;
        float[] xs = new float[chunk * sample];
        int[] ys = new int[chunk];
        double accuracy = 0;
        for (int b = 0; b < evalCount / chunk; b++) {
            test.batch(b, chunk, xs, ys);
            Tensor input = Tensor.of(xs, chunk, shape[0], shape[1], shape[2]);
            accuracy += F.accuracy(ai.beestream.jitnn.Grad.inference(() -> model.forward(input)), Tensor.of(ys, chunk)) * chunk / evalCount;
        }

        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(reference)).order(ByteOrder.LITTLE_ENDIAN);
        float torchAccuracy = buf.getFloat();
        float[] torchLosses = new float[buf.remaining() / 4];
        buf.asFloatBuffer().get(torchLosses);
        System.out.printf("[jitnn] %s first losses %s vs PyTorch %.5f %.5f %.5f %.5f%n", name, losses.subList(0, 4), torchLosses[0], torchLosses[1], torchLosses[2], torchLosses[3]);
        assertTrue(torchLosses.length == steps, name + ": PyTorch ran " + torchLosses.length + " steps, jitNN " + steps);
        double first = 0;
        for (int i = 0; i < 10; i++) {
            first = Math.max(first, Math.abs(losses.get(i) - torchLosses[i]) / Math.abs(torchLosses[i]));
        }
        double meanJ = 0;
        double meanT = 0;
        int tail = Math.min(50, steps / 4);
        for (int i = steps - tail; i < steps; i++) {
            meanJ += losses.get(i) / tail;
            meanT += torchLosses[i] / tail;
        }
        System.out.printf("[jitnn] %s: %d params, first step %.1f s, then %.1f ms/step (%.0f samples/s), %d plans compiled; vs PyTorch: steps 0-9 gap %.2e, "
                + "mean loss of the last 50 steps %.4f vs %.4f, eval accuracy %.1f%% vs %.1f%%%n", name, model.parameterCount(), firstStep / 1e9, steady,
                BATCH * 1000 / steady, plans1[0] - plans0[0], first, meanJ, meanT, 100 * accuracy, 100 * torchAccuracy);
        // Step 0 is deterministic; after it, Adam's sign-like first updates amplify rounding noise
        // in near-zero gradients, so the curves are compared by their end, not step by step.
        assertTrue(Math.abs(losses.get(0) - torchLosses[0]) <= 1e-5 * torchLosses[0], name + ": step-0 loss " + losses.get(0) + " vs " + torchLosses[0]);
        assertTrue(Math.abs(meanJ - meanT) <= 0.1 * meanT + 0.02, name + ": final mean loss " + meanJ + " vs " + meanT);
        assertTrue(Math.abs(accuracy - torchAccuracy) <= 0.03, name + ": accuracy " + accuracy + " vs PyTorch " + torchAccuracy);
    }
}
