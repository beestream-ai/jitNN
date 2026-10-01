package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.models.MobileNetV2;
import org.beehive.jitnn.nn.models.ResNet;
import org.beehive.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Wave 2 gates: ResNet-18 and MobileNetV2 (small variants) for 200 MNIST steps against PyTorch
 * from the same initial weights (scripts/pytorch/cnn_train.py): losses step by step, then
 * accuracy in evaluation mode (running BatchNorm statistics) on the first 1000 test images.
 */
class CnnAccelTest {

    private static final int STEPS = 200;
    private static final int BATCH = 64;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void resNet18MatchesPyTorch() throws Exception {
        run("ResNet-18", ResNet.resnet18(1, 10, true, 1), "resnet18_pytorch.f32");
    }

    @Test
    void mobileNetV2MatchesPyTorch() throws Exception {
        run("MobileNetV2", MobileNetV2.mobilenetV2(1, 10, true, 0, 1), "mobilenetv2_pytorch.f32");
    }

    /** Every parameter's step-0 gradient against PyTorch's, as a relative L2 error. */
    private static void checkGradients(String name, Module model, Path file) throws Exception {
        assumeTrue(Files.exists(file), "missing " + file);
        java.nio.FloatBuffer ref = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        double worst = 0;
        int worstAt = -1;
        int index = 0;
        for (Tensor p : model.parameters()) {
            float[] g = p.grad().toFloatArray();
            double diff = 0;
            double norm = 0;
            for (float v : g) {
                double t = ref.get();
                diff += (v - t) * (v - t);
                norm += t * t;
            }
            double rel = Math.sqrt(diff) / Math.max(Math.sqrt(norm), 1e-12);
            if (rel > worst) {
                worst = rel;
                worstAt = index;
            }
            index++;
        }
        assertTrue(!ref.hasRemaining(), name + ": PyTorch has more gradient values than jitNN parameters");
        System.out.printf("[jitnn] %s step-0 gradients of %d parameters: worst relative L2 error %.2e (parameter %d)%n", name, index, worst, worstAt);
        assertTrue(worst < 1e-3, name + ": step-0 gradient of parameter " + worstAt + " differs by " + worst);
    }

    private static void run(String name, Module model, String referenceFile) throws Exception {
        String dir = System.getProperty("jitnn.mnist", "");
        assumeTrue(!dir.isEmpty() && Mnist.isPresent(Path.of(dir)), "set -Djitnn.mnist=<dir>");
        Path reference = Path.of(dir, referenceFile);
        assumeTrue(Files.exists(reference), "run scripts/pytorch/cnn_train.py to write " + reference);
        Mnist train = Mnist.load(Path.of(dir), true);
        Mnist test = Mnist.load(Path.of(dir), false);
        Adam opt = new Adam(model.parameters(), 1e-3f);
        float[] xb = new float[BATCH * Mnist.PIXELS];
        int[] yb = new int[BATCH];
        List<Float> losses = new ArrayList<>();
        long[] plans0 = Engine.planStats();
        long t0 = System.nanoTime();
        long firstStep = 0;
        for (int b = 0; b < STEPS; b++) {
            train.batch(b, BATCH, xb, yb);
            Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, BATCH, 1, 28, 28)), Tensor.of(yb, BATCH));
            loss.backward();
            if (b == 0) {
                checkGradients(name, model, Path.of(dir, referenceFile + ".grad0"));
            }
            opt.step();
            losses.add(loss.item());
            opt.zeroGrad();
            if (b == 0) {
                firstStep = System.nanoTime() - t0;
            }
        }
        double steady = (System.nanoTime() - t0 - firstStep) / 1e6 / (STEPS - 1);
        long[] plans1 = Engine.planStats();

        model.train(false);
        float[] xe = new float[1000 * Mnist.PIXELS];
        int[] ye = new int[1000];
        test.batch(0, 1000, xe, ye);
        double accuracy = 0;
        for (int b = 0; b < 4; b++) {
            float[] xs = new float[250 * Mnist.PIXELS];
            int[] ys = new int[250];
            System.arraycopy(xe, b * 250 * Mnist.PIXELS, xs, 0, xs.length);
            System.arraycopy(ye, b * 250, ys, 0, 250);
            accuracy += F.accuracy(model.forward(Tensor.of(xs, 250, 1, 28, 28)), Tensor.of(ys, 250)) / 4;
        }

        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(reference)).order(ByteOrder.LITTLE_ENDIAN);
        float torchAccuracy = buf.getFloat();
        float[] torchLosses = new float[buf.remaining() / 4];
        buf.asFloatBuffer().get(torchLosses);
        System.out.printf("[jitnn] %s first losses %s vs PyTorch %.5f %.5f %.5f %.5f%n", name, losses.subList(0, 4), torchLosses[0], torchLosses[1], torchLosses[2], torchLosses[3]);
        double first = 0;
        for (int i = 0; i < 10; i++) {
            first = Math.max(first, Math.abs(losses.get(i) - torchLosses[i]) / Math.abs(torchLosses[i]));
        }
        double meanJ = 0;
        double meanT = 0;
        for (int i = STEPS - 50; i < STEPS; i++) {
            meanJ += losses.get(i) / 50.0;
            meanT += torchLosses[i] / 50.0;
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
