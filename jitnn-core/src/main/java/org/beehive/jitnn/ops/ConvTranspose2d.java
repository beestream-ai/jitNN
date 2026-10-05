package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;

/**
 * Transposed 2-D convolution, PyTorch semantics and layout: x [N, Cin, H, W], weight
 * [Cin, Cout/groups, K, K], square kernel, symmetric stride, padding, output padding and dilation;
 * output [N, Cout, OH, OW] with OH = (H - 1) stride - 2 pad + dil (K - 1) + outPad + 1.
 *
 * <p>It is the data gradient of the convolution that maps [N, Cout, OH, OW] to [N, Cin, H, W]
 * with the same weight, so it runs on the convolution kernels: forward is that convolution's
 * data gradient, the data gradient is its forward pass and the weight gradient its weight
 * gradient with the roles of input and output swapped. iattr holds that convolution's
 * attributes (see {@link Conv2d}).
 */
public final class ConvTranspose2d implements Op {

    static final ConvTranspose2d OP = new ConvTranspose2d();

    private ConvTranspose2d() {
    }

    public static Tensor apply(Tensor x, Tensor w, Tensor bias, int stride, int pad, int outPad, int dil, int groups) {
        Checks.f32(x, w);
        Checks.rank(x, 4);
        Checks.rank(w, 4);
        int n = x.dim(0);
        int cin = x.dim(1);
        int h = x.dim(2);
        int wd = x.dim(3);
        int k = w.dim(2);
        int cout = w.dim(1) * groups;
        if (w.dim(3) != k) {
            throw new IllegalArgumentException("conv_transpose2d: only square kernels are supported, got " + w.shape());
        }
        if (groups < 1 || cin % groups != 0 || w.dim(0) != cin) {
            throw new IllegalArgumentException("conv_transpose2d: input " + x.shape() + ", weight " + w.shape() + " and groups " + groups + " do not match");
        }
        if (stride < 1 || pad < 0 || dil < 1 || outPad < 0 || (outPad >= stride & outPad >= dil)) {
            throw new IllegalArgumentException("conv_transpose2d: bad stride/pad/outPad/dilation " + stride + "/" + pad + "/" + outPad + "/" + dil);
        }
        int oh = (h - 1) * stride - 2 * pad + dil * (k - 1) + outPad + 1;
        int ow = (wd - 1) * stride - 2 * pad + dil * (k - 1) + outPad + 1;
        if (oh < 1 || ow < 1) {
            throw new IllegalArgumentException("conv_transpose2d: output would be empty for input " + x.shape());
        }
        // the convolution [N, cout, oh, ow] -> [N, cin, h, w] whose data gradient this is
        int[] a = { n, cout, oh, ow, cin, h, wd, k, stride, pad, dil, groups, 0, 0 };
        Tensor out = Tensor.result(Shape.of(n, cout, oh, ow), DType.F32);
        Tensor.record(OP, new Tensor[] { x, w }, new Tensor[] { out }, a, null, true);
        if (bias == null) {
            return out;
        }
        Checks.f32(bias);
        if (bias.numel() != cout) {
            throw new IllegalArgumentException("conv_transpose2d: bias " + bias.shape() + " for " + cout + " channels");
        }
        return out.add(bias.reshape(1, cout, 1, 1));
    }

    @Override
    public String name() {
        return "conv_transpose2d";
    }

    @Override
    public void emit(Emitter e, Node node) {
        Conv2d.BACKWARD_DATA.emit(e, node);
    }

    @Override
    public void reference(Node node) {
        Conv2d.BACKWARD_DATA.reference(node);
    }

    @Override
    public Tensor[] backward(Node node, Tensor gy) {
        int[] a = node.iattr;
        Tensor x = node.in(0);
        Tensor w = node.in(1);
        Tensor gx = null;
        Tensor gw = null;
        if (x.requiresGrad()) {
            gx = Tensor.result(x.shape(), DType.F32);
            Tensor.record(Conv2d.OP, new Tensor[] { gy, w }, new Tensor[] { gx }, a.clone(), null, true);
        }
        if (w.requiresGrad()) {
            gw = Tensor.result(w.shape(), DType.F32);
            Tensor.record(Conv2d.BACKWARD_WEIGHT, new Tensor[] { x, gy }, new Tensor[] { gw }, a.clone(), null, true);
        }
        return new Tensor[] { gx, gw };
    }
}
