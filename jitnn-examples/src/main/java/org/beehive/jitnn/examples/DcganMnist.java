package org.beehive.jitnn.examples;

import java.awt.image.BufferedImage;
import java.nio.file.Path;

import javax.imageio.ImageIO;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.DataLoader;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.nn.BatchNorm2d;
import org.beehive.jitnn.nn.Conv2d;
import org.beehive.jitnn.nn.ConvTranspose2d;
import org.beehive.jitnn.nn.LeakyReLU;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.ReLU;
import org.beehive.jitnn.nn.Sequential;
import org.beehive.jitnn.nn.Tanh;
import org.beehive.jitnn.nn.optim.Adam;

/**
 * A generative adversarial network that learns to draw digits, PyTorch's DCGAN example with its
 * settings ({@code --dataset mnist}): MNIST resized to 64x64, a generator of five transposed
 * convolutions (512-256-128-64 channels) against a discriminator of five convolutions, Adam (lr
 * 2e-4, beta1 0.5) for both, batches of 64, 25 epochs. Afterwards it draws 64 digits into
 * dcgan-digits.png.
 *
 * <pre>run.sh DcganMnist [epochs]</pre>
 */
public final class DcganMnist {

    /** Records both steps before reading a loss, so each training step is one compiled plan ({@code -Djitnn.examples.dcgan.oneTrace}). */
    static final boolean ONE_TRACE = Boolean.getBoolean("jitnn.examples.dcgan.oneTrace");

    static final int NZ = 100;
    static final int NGF = 64;
    static final int NDF = 64;

    static Module generator() {
        return new Sequential(
                new ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, 0, 1, 1, false, 1), new BatchNorm2d(NGF * 8), new ReLU(),          // 4x4
                new ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, 0, 1, 1, false, 2), new BatchNorm2d(NGF * 4), new ReLU(),     // 8x8
                new ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, 0, 1, 1, false, 3), new BatchNorm2d(NGF * 2), new ReLU(),     // 16x16
                new ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, 0, 1, 1, false, 4), new BatchNorm2d(NGF), new ReLU(),             // 32x32
                new ConvTranspose2d(NGF, 1, 4, 2, 1, 0, 1, 1, false, 5), new Tanh());                                       // 64x64
    }

    /** One logit per image; the original's sigmoid and BCELoss are BCE with logits. */
    static Module discriminator() {
        return new Sequential(
                new Conv2d(1, NDF, 4, 2, 1, 1, 1, false, 11), new LeakyReLU(0.2f),                                          // 32x32
                new Conv2d(NDF, NDF * 2, 4, 2, 1, 1, 1, false, 12), new BatchNorm2d(NDF * 2), new LeakyReLU(0.2f),          // 16x16
                new Conv2d(NDF * 2, NDF * 4, 4, 2, 1, 1, 1, false, 13), new BatchNorm2d(NDF * 4), new LeakyReLU(0.2f),      // 8x8
                new Conv2d(NDF * 4, NDF * 8, 4, 2, 1, 1, 1, false, 14), new BatchNorm2d(NDF * 8), new LeakyReLU(0.2f),      // 4x4
                new Conv2d(NDF * 8, 1, 4, 1, 0, 1, 1, false, 15));                                                         // 1x1
    }

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 25);
        int batch = 64;
        Mnist train = Mnist.load(Examples.mnist(), true, true);

        Module g = generator();
        Module d = discriminator();
        Adam optG = new Adam(g.parameters(), 2e-4f, 0.5f, 0.999f, 1e-8f, 0, false);
        Adam optD = new Adam(d.parameters(), 2e-4f, 0.5f, 0.999f, 1e-8f, 0, false);
        System.out.printf("generator %,d parameters, discriminator %,d%n", g.parameterCount(), d.parameterCount());

        int step = 0;
        try (DataLoader loader = DataLoader.builder(train, batch).shuffle(42).build()) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                long t0 = System.nanoTime();
                float sumD = 0;
                float sumG = 0;
                for (DataLoader.Batch b : loader.epoch(epoch)) {
                    int n = b.size();
                    // to 64x64 in [-1, 1], as the original's Resize(64) and Normalize(0.5, 0.5)
                    Tensor real = F.interpolate(b.x().reshape(n, 1, 28, 28), 64, 64, true, false).mul(2).add(-1);
                    Tensor ones = Tensor.of(fill(n, 1), n, 1, 1, 1);
                    Tensor zeros = Tensor.of(fill(n, 0), n, 1, 1, 1);
                    Tensor z = Tensor.normal(Tensor.seed(step++), 0, 1, n, NZ, 1, 1);

                    // discriminator: real images towards 1, generated ones towards 0
                    Tensor fake = g.forward(z);
                    Tensor lossD = F.binaryCrossEntropyWithLogits(d.forward(real), ones)
                            .add(F.binaryCrossEntropyWithLogits(d.forward(fake.detach()), zeros));
                    lossD.backward();
                    optD.step();
                    if (!ONE_TRACE) {
                        sumD += lossD.item();      // a sync point: the generator's step becomes a second plan
                    }
                    optD.zeroGrad();

                    // generator: make the updated discriminator say 1 for the same fakes
                    Tensor lossG = F.binaryCrossEntropyWithLogits(d.forward(fake), ones);
                    lossG.backward();
                    optG.step();
                    if (ONE_TRACE) {
                        sumD += lossD.item();      // both steps were recorded first: one plan per step
                    }
                    sumG += lossG.item();
                    optG.zeroGrad();
                    d.zeroGrad();
                }
                System.out.printf("epoch %d: discriminator loss %.3f, generator loss %.3f, %.2f s%n", epoch + 1, sumD / loader.batches(), sumG / loader.batches(),
                        Examples.seconds(t0));
            }
        }

        g.train(false);
        Tensor z = Tensor.normal(7, 0, 1, 64, NZ, 1, 1);
        float[] digits = Grad.inference(() -> g.forward(z)).toFloatArray();
        BufferedImage img = new BufferedImage(8 * 64, 8 * 64, BufferedImage.TYPE_BYTE_GRAY);
        for (int k = 0; k < 64; k++) {
            for (int p = 0; p < 64 * 64; p++) {
                int v = Math.round(255 * Math.min(1, Math.max(0, (digits[k * 4096 + p] + 1) / 2)));
                img.getRaster().setSample((k % 8) * 64 + p % 64, (k / 8) * 64 + p / 64, 0, v);
            }
        }
        Path png = Path.of("dcgan-digits.png");
        ImageIO.write(img, "png", png.toFile());
        System.out.println("wrote 64 generated digits to " + png.toAbsolutePath());
    }

    private static float[] fill(int n, float v) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, v);
        return a;
    }
}
