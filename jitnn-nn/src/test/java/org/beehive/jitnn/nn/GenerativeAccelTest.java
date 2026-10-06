package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.models.Dcgan;
import org.beehive.jitnn.nn.models.UNet;
import org.beehive.jitnn.nn.models.Vae;
import org.beehive.jitnn.nn.optim.Adam;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Wave 4 gates on MNIST ({@code -Djitnn.mnist=<dir>} with the PyTorch references of
 * scripts/pytorch/wave4.py): a VAE (BCE plus Gaussian KL, reparameterised with fixed noise), a
 * DCGAN (transposed convolutions, LeakyReLU, BCE with logits, two optimisers) and a U-Net
 * (transposed convolution, bilinear upsampling, skip concatenation). Same initial weights, batches
 * and noise as PyTorch; checks the step-0 gradients against FP64, the loss curves and the
 * evaluation metrics.
 */
class GenerativeAccelTest {

    // The same constants as wave4.py.
    static final int VAE_BATCH = 128, VAE_STEPS = 400, VAE_HIDDEN = 400, VAE_LATENT = 20, VAE_NOISE = 5000, VAE_EVAL_NOISE = 9000, VAE_EVAL_BATCHES = 10;
    static final float VAE_LR = 1e-3f;
    static final int GAN_BATCH = 64, GAN_STEPS = 300, GAN_LATENT = 64, GAN_WIDTH = 32, GAN_NOISE = 6000, GAN_G_SEED = 1, GAN_D_SEED = 50;
    static final float GAN_LR = 2e-4f, GAN_BETA1 = 0.5f;
    static final int UNET_BATCH = 32, UNET_STEPS = 300, UNET_WIDTH = 16, UNET_EVAL_BATCHES = 32;
    static final float UNET_LR = 1e-3f;
    static final float SQRT3 = (float) Math.sqrt(3.0);

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static Path dir() {
        String d = System.getProperty("jitnn.mnist", "");
        assumeTrue(!d.isEmpty() && Mnist.isPresent(Path.of(d)), "set -Djitnn.mnist=<dir>");
        return Path.of(d);
    }

    private static float[] batch(Mnist m, int b, int size) {
        int n = m.count / size;
        float[] x = new float[size * Mnist.PIXELS];
        System.arraycopy(m.images, (b % n) * size * Mnist.PIXELS, x, 0, x.length);
        return x;
    }

    private static float[] foreground(float[] unit) {
        float[] t = new float[unit.length];
        for (int i = 0; i < t.length; i++) {
            t[i] = unit[i] > 127.0f / 255.0f ? 1.0f : 0.0f;
        }
        return t;
    }

    private static void timing(String name, long t0, long firstStep, int steps, Module... models) {
        long params = 0;
        for (Module m : models) {
            params += m.parameterCount();
        }
        double steady = (System.nanoTime() - t0 - firstStep) / 1e6 / (steps - 1);
        System.out.printf("[jitnn] %s: %d params, first step %.1f s, then %.1f ms/step%n", name, params, firstStep / 1e9, steady);
    }

    @Test
    void vaeMatchesPyTorch() throws Exception {
        Path dir = dir();
        Path reference = dir.resolve("vae_pytorch.f32");
        Curves torch = Curves.load(reference, 1, "scripts/pytorch/wave4.py train vae");
        Mnist train = Mnist.load(dir, true, true);
        Mnist test = Mnist.load(dir, false, true);
        Vae model = new Vae(Mnist.PIXELS, VAE_HIDDEN, VAE_LATENT, 1);
        Adam opt = new Adam(model.parameters(), VAE_LR);
        List<Float> losses = new ArrayList<>();
        long t0 = System.nanoTime();
        long first = 0;
        for (int b = 0; b < VAE_STEPS; b++) {
            Tensor x = Tensor.of(batch(train, b, VAE_BATCH), VAE_BATCH, Mnist.PIXELS);
            Tensor eps = Tensor.uniform(Tensor.seed(VAE_NOISE + b), -SQRT3, SQRT3, VAE_BATCH, VAE_LATENT);
            Tensor loss = Vae.loss(model.run(x, eps), x);
            opt.zeroGrad();
            loss.backward();
            if (b == 0) {
                Gates.checkGradients("VAE", model.parameters(), reference.toString());
            }
            opt.step();
            losses.add(loss.item());
            if (b == 0) {
                first = System.nanoTime() - t0;
            }
        }
        timing("VAE", t0, first, VAE_STEPS, model);
        double eval = 0;
        for (int i = 0; i < VAE_EVAL_BATCHES; i++) {
            Tensor x = Tensor.of(batch(test, i, VAE_BATCH), VAE_BATCH, Mnist.PIXELS);
            Tensor eps = Tensor.uniform(Tensor.seed(VAE_EVAL_NOISE + i), -SQRT3, SQRT3, VAE_BATCH, VAE_LATENT);
            eval += Grad.inference(() -> Vae.loss(model.run(x, eps), x)).item() / VAE_EVAL_BATCHES;
        }
        System.out.printf("[jitnn] VAE test loss %.3f vs PyTorch %.3f%n", eval, torch.metrics()[0]);
        Curves.check("VAE", Curves.series(losses, 0, 1), torch.series(0, 1), 50, 1e-5, 0.02);
        assertTrue(Math.abs(eval - torch.metrics()[0]) <= 0.02 * torch.metrics()[0], "VAE test loss " + eval + " vs " + torch.metrics()[0]);
    }

    @Test
    void dcganMatchesPyTorch() throws Exception {
        Path dir = dir();
        Path reference = dir.resolve("dcgan_pytorch.f32");
        Curves torch = Curves.load(reference, 0, "scripts/pytorch/wave4.py train dcgan");
        Mnist train = Mnist.load(dir, true, true);
        Dcgan.Generator g = new Dcgan.Generator(GAN_LATENT, GAN_WIDTH, GAN_G_SEED);
        Dcgan.Discriminator d = new Dcgan.Discriminator(GAN_WIDTH, GAN_D_SEED);
        Adam gOpt = new Adam(g.parameters(), GAN_LR, GAN_BETA1, 0.999f, 1e-8f, 0.0f, false);
        Adam dOpt = new Adam(d.parameters(), GAN_LR, GAN_BETA1, 0.999f, 1e-8f, 0.0f, false);
        Tensor ones = Tensor.ones(GAN_BATCH, 1);
        Tensor zeros = Tensor.zeros(GAN_BATCH, 1);
        List<Float> losses = new ArrayList<>();
        long t0 = System.nanoTime();
        long first = 0;
        for (int b = 0; b < GAN_STEPS; b++) {
            Tensor real = Tensor.of(batch(train, b, GAN_BATCH), GAN_BATCH, 1, 28, 28).mul(2.0f).add(-1.0f);
            Tensor z = Tensor.uniform(Tensor.seed(GAN_NOISE + b), -SQRT3, SQRT3, GAN_BATCH, GAN_LATENT);
            Tensor fake = g.forward(z);
            Tensor dLoss = F.binaryCrossEntropyWithLogits(d.forward(real), ones).add(F.binaryCrossEntropyWithLogits(d.forward(fake.detach()), zeros));
            dOpt.zeroGrad();
            dLoss.backward();
            if (b == 0) {
                Gates.checkGradients("DCGAN discriminator", d.parameters(), reference + "_d");
            }
            dOpt.step();
            Tensor gLoss = F.binaryCrossEntropyWithLogits(d.forward(fake), ones);
            gOpt.zeroGrad();
            gLoss.backward();
            if (b == 0) {
                Gates.checkGradients("DCGAN generator", g.parameters(), reference + "_g");
            }
            gOpt.step();
            losses.add(dLoss.item());
            losses.add(gLoss.item());
            if (b == 0) {
                first = System.nanoTime() - t0;
            }
        }
        timing("DCGAN", t0, first, GAN_STEPS, g, d);
        // Adversarial training amplifies rounding differences; the curves are compared on their tails.
        Curves.check("DCGAN discriminator", Curves.series(losses, 0, 2), torch.series(0, 2), 50, 1e-5, 0.10);
        Curves.check("DCGAN generator", Curves.series(losses, 1, 2), torch.series(1, 2), 50, 1e-5, 0.10);
    }

    @Test
    void unetMatchesPyTorch() throws Exception {
        Path dir = dir();
        Path reference = dir.resolve("unet_pytorch.f32");
        Curves torch = Curves.load(reference, 2, "scripts/pytorch/wave4.py train unet");
        Mnist train = Mnist.load(dir, true, true);
        Mnist test = Mnist.load(dir, false, true);
        UNet model = new UNet(1, UNET_WIDTH, 1);
        Adam opt = new Adam(model.parameters(), UNET_LR);
        List<Float> losses = new ArrayList<>();
        long t0 = System.nanoTime();
        long first = 0;
        for (int b = 0; b < UNET_STEPS; b++) {
            float[] x = batch(train, b, UNET_BATCH);
            Tensor loss = F.binaryCrossEntropyWithLogits(model.forward(Tensor.of(x, UNET_BATCH, 1, 28, 28)), Tensor.of(foreground(x), UNET_BATCH, 1, 28, 28));
            opt.zeroGrad();
            loss.backward();
            if (b == 0) {
                Gates.checkGradients("U-Net", model.parameters(), reference.toString());
            }
            opt.step();
            losses.add(loss.item());
            if (b == 0) {
                first = System.nanoTime() - t0;
            }
        }
        timing("U-Net", t0, first, UNET_STEPS, model);
        model.train(false);
        double eval = 0;
        double inter = 0;
        double union = 0;
        for (int i = 0; i < UNET_EVAL_BATCHES; i++) {
            float[] x = batch(test, i, UNET_BATCH);
            float[] target = foreground(x);
            Tensor t = Tensor.of(target, UNET_BATCH, 1, 28, 28);
            Tensor logits = Grad.inference(() -> model.forward(Tensor.of(x, UNET_BATCH, 1, 28, 28)));
            eval += F.binaryCrossEntropyWithLogits(logits, t).item() / UNET_EVAL_BATCHES;
            float[] l = logits.toFloatArray();
            for (int k = 0; k < l.length; k++) {
                boolean p = l[k] > 0;
                inter += p && target[k] > 0 ? 1 : 0;
                union += p || target[k] > 0 ? 1 : 0;
            }
        }
        double iou = inter / union;
        System.out.printf("[jitnn] U-Net test loss %.4f vs PyTorch %.4f, foreground IoU %.4f vs %.4f%n", eval, torch.metrics()[0], iou, torch.metrics()[1]);
        // The loss ends near 0.02, where FP32 summation order alone (split-K on or off) moves the
        // late curve by 5-7%; the step-0 gradients above are the strict check.
        Curves.check("U-Net", Curves.series(losses, 0, 1), torch.series(0, 1), 50, 1e-5, 0.10);
        assertTrue(Math.abs(eval - torch.metrics()[0]) <= 0.10 * torch.metrics()[0], "U-Net test loss " + eval + " vs " + torch.metrics()[0]);
        assertTrue(Math.abs(iou - torch.metrics()[1]) <= 0.01, "U-Net IoU " + iou + " vs " + torch.metrics()[1]);
    }
}
