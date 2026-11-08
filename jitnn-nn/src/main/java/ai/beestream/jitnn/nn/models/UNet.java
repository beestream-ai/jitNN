package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Conv2d;
import ai.beestream.jitnn.nn.ConvTranspose2d;
import ai.beestream.jitnn.nn.Module;

/**
 * A small U-Net (Ronneberger et al., 2015) for 28x28 inputs with two down-sampling levels:
 * double 3x3 convolutions with ReLU, max pooling, a bottleneck, then one level up through a
 * stride-2 transposed convolution and one through bilinear upsampling and a 1x1 convolution,
 * each concatenated with the encoder's features of that size; a 1x1 convolution gives one logit
 * per pixel. Convolutions are seeded in construction order (encoder, bottleneck, decoder, head).
 */
public final class UNet extends Module {

    private final Conv2d e1a, e1b, e2a, e2b, ba, bb, d2a, d2b, up1, d1a, d1b, head;
    private final ConvTranspose2d up2;

    public UNet(int in, int width, long seed) {
        Seeds s = new Seeds(seed);
        int w = width;
        e1a = conv(in, w, 3, s);
        e1b = conv(w, w, 3, s);
        e2a = conv(w, 2 * w, 3, s);
        e2b = conv(2 * w, 2 * w, 3, s);
        ba = conv(2 * w, 4 * w, 3, s);
        bb = conv(4 * w, 4 * w, 3, s);
        up2 = child(new ConvTranspose2d(4 * w, 2 * w, 2, 2, 0, 0, 1, 1, true, s.next()));
        d2a = conv(4 * w, 2 * w, 3, s);
        d2b = conv(2 * w, 2 * w, 3, s);
        up1 = conv(2 * w, w, 1, s);
        d1a = conv(2 * w, w, 3, s);
        d1b = conv(w, w, 3, s);
        head = conv(w, 1, 1, s);
    }

    private Conv2d conv(int in, int out, int k, Seeds s) {
        return child(new Conv2d(in, out, k, 1, k / 2, 1, 1, true, s.next()));
    }

    /** Logits [B, 1, H, W] for x [B, in, H, W] (H and W divisible by 4). */
    @Override
    public Tensor forward(Tensor x) {
        Tensor f1 = e1b.forwardRelu(e1a.forwardRelu(x));
        Tensor f2 = e2b.forwardRelu(e2a.forwardRelu(F.maxPool2d(f1, 2, 2, 0)));
        Tensor b = bb.forwardRelu(ba.forwardRelu(F.maxPool2d(f2, 2, 2, 0)));
        Tensor u2 = d2b.forwardRelu(d2a.forwardRelu(Tensor.cat(1, up2.forward(b), f2)));
        Tensor u1 = up1.forward(F.upsampleBilinear(u2, 2, false));
        Tensor y = d1b.forwardRelu(d1a.forwardRelu(Tensor.cat(1, u1, f1)));
        return head.forward(y);
    }
}
