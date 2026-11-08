package org.beehive.jitnn.examples;

import java.awt.image.BufferedImage;
import java.nio.file.Path;

import javax.imageio.ImageIO;

import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.DataLoader;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.nn.models.Vae;
import org.beehive.jitnn.nn.optim.Adam;

/**
 * A variational autoencoder that learns to draw digits, PyTorch's VAE example with its settings
 * (also DL4J's VariationalAutoEncoderExample): 784-400-20 encoder and decoder, the negative ELBO,
 * Adam (lr 1e-3), batches of 128, 10 epochs, the test loss after every epoch. Afterwards it
 * decodes 64 random points of the latent space into vae-digits.png.
 *
 * <pre>run.sh VaeMnist [epochs]</pre>
 */
public final class VaeMnist {

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 10);
        int batch = 128;
        int latent = 20;
        Mnist train = Mnist.load(Examples.mnist(), true, true);    // pixels in [0, 1], the decoder's targets
        Mnist test = Mnist.load(Examples.mnist(), false, true);

        Vae vae = new Vae(784, 400, latent, 1);
        Adam opt = new Adam(vae.parameters(), 1e-3f);
        int step = 0;
        try (DataLoader loader = DataLoader.builder(train, batch).shuffle(42).build();
                DataLoader testLoader = DataLoader.builder(test, batch).build()) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                long t0 = System.nanoTime();
                float sum = 0;
                for (DataLoader.Batch b : loader.epoch(epoch)) {
                    Tensor x = b.x().reshape(b.size(), 784);
                    // fresh noise every step; a seed tensor keeps the compiled step replayable
                    Tensor eps = Tensor.normal(Tensor.seed(step++), 0, 1, b.size(), latent);
                    Tensor loss = Vae.loss(vae.run(x, eps), x);
                    loss.backward();
                    opt.step();
                    sum += loss.item() * b.size();
                    opt.zeroGrad();
                }
                vae.train(false);
                float testSum = 0;
                for (DataLoader.Batch b : testLoader.epoch(0)) {
                    Tensor x = b.x().reshape(b.size(), 784);
                    Tensor eps = Tensor.normal(Tensor.seed(step++), 0, 1, b.size(), latent);
                    testSum += Grad.inference(() -> Vae.loss(vae.run(x, eps), x)).item() * b.size();
                }
                vae.train(true);
                System.out.printf("epoch %d: loss %.1f per image, test loss %.1f, %.2f s%n", epoch + 1, sum / train.count(), testSum / test.count(), Examples.seconds(t0));
            }
        }

        vae.train(false);
        Tensor z = Tensor.normal(7, 0, 1, 64, latent);
        float[] digits = Grad.inference(() -> vae.decode(z)).toFloatArray();
        BufferedImage img = new BufferedImage(8 * 28, 8 * 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int k = 0; k < 64; k++) {
            for (int p = 0; p < 784; p++) {
                int v = Math.round(255 * Math.min(1, Math.max(0, digits[k * 784 + p])));
                img.getRaster().setSample((k % 8) * 28 + p % 28, (k / 8) * 28 + p / 28, 0, v);
            }
        }
        Path png = Path.of("vae-digits.png");
        ImageIO.write(img, "png", png.toFile());
        System.out.println("wrote 64 generated digits to " + png.toAbsolutePath());
    }
}
