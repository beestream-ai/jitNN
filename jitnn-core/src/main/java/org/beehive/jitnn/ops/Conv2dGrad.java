package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;

/** Direct access to the convolution gradients, for benchmarks and tests of the backward kernels. */
public final class Conv2dGrad {

    private Conv2dGrad() {
    }

    private static int[] attrs(Shape x, Shape w, int stride, int pad, int dil, int groups) {
        int k = w.dim(2);
        int oh = (x.dim(2) + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        int ow = (x.dim(3) + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        return new int[] { x.dim(0), x.dim(1), x.dim(2), x.dim(3), w.dim(0), oh, ow, k, stride, pad, dil, groups, 0, 0 };
    }

    /** dx of a convolution with input shape {@code xShape}, given dy and the weight. */
    public static Tensor input(Tensor dy, Tensor w, Shape xShape, int stride, int pad, int dil, int groups) {
        Tensor dx = Tensor.result(xShape, DType.F32);
        Tensor.record(Conv2d.BACKWARD_DATA, new Tensor[] { dy, w }, new Tensor[] { dx }, attrs(xShape, w.shape(), stride, pad, dil, groups), null, true);
        return dx;
    }

    /** dW of a convolution with weight shape {@code wShape}, given dy and the input. */
    public static Tensor weight(Tensor dy, Tensor x, Shape wShape, int stride, int pad, int dil, int groups) {
        Tensor dw = Tensor.result(wShape, DType.F32);
        Tensor.record(Conv2d.BACKWARD_WEIGHT, new Tensor[] { dy, x }, new Tensor[] { dw }, attrs(x.shape(), wShape, stride, pad, dil, groups), null, true);
        return dw;
    }
}
