package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.FftKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * One discrete Fourier transform along the last axis (see {@link FftKernels#dft}); its gradient is
 * the adjoint transform, another Dft. iattr = [inStride, nIn, nOut, n, sign (+1/-1), realIn,
 * realOut, weight]; fattr = [scale].
 */
public final class Dft implements Op {

    static final Dft OP = new Dft();

    private Dft() {
    }

    /**
     * x [..., stride] (realIn) or [..., stride, 2]; transforms the first nIn entries of each row with
     * length n into nOut outputs.
     */
    public static Tensor apply(Tensor x, int nIn, int nOut, int n, int sign, float scale, boolean realIn, boolean realOut, int weight) {
        Checks.f32(x);
        int lead = x.rank() - (realIn ? 1 : 2);
        if (lead < 0 || (!realIn && x.dim(-1) != 2)) {
            throw new IllegalArgumentException("DFT needs " + (realIn ? "a real tensor" : "complex pairs [..., n, 2]") + ", got " + x.shape());
        }
        int stride = x.dim(lead);
        int[] dims = java.util.Arrays.copyOf(x.shape().dims(), lead + (realOut ? 1 : 2));
        dims[lead] = nOut;
        if (!realOut) {
            dims[lead + 1] = 2;
        }
        Tensor y = Tensor.result(Shape.of(dims), DType.F32);
        float[] tw = new float[2 * n];
        for (int p = 0; p < n; p++) {
            double ang = sign * 2 * Math.PI * p / n;
            tw[2 * p] = (float) Math.cos(ang);
            tw[2 * p + 1] = (float) Math.sin(ang);
        }
        Tensor.record(OP, new Tensor[] { x, Tensor.of(tw, n, 2) }, new Tensor[] { y }, new int[] { stride, nIn, nOut, n, sign, realIn ? 1 : 0, realOut ? 1 : 0, weight }, new float[] { scale }, true);
        return y;
    }

    private static int rows(Node node) {
        int[] a = node.iattr;
        return (int) (node.out(0).shape().numel() / (a[2] * (a[6] == 1 ? 1 : 2)));
    }

    @Override
    public String name() {
        return "dft";
    }

    @Override
    public void emit(Emitter e, Node node) {
        FloatArray x = e.f32(node.in(0));
        FloatArray tw = e.f32(node.in(1));
        FloatArray y = e.f32(node.out(0));
        int[] a = node.iattr;
        int stride = a[0], nIn = a[1], nOut = a[2], n = a[3], realIn = a[5], realOut = a[6], weight = a[7];
        float scale = node.fattr[0];
        int total = rows(node) * nOut;
        // whole rows, full length, no Hermitian weighting: one batched cuFFT call
        boolean c2c = realIn == 0 && realOut == 0 && nOut == n;
        boolean r2c = realIn == 1 && realOut == 0 && a[4] < 0 && nOut == n / 2 + 1;
        if (nIn == n && stride == nIn && weight == 0 && (c2c || r2c) && org.beehive.jitnn.natives.Libs.fft(e, x, y, n, rows(node), a[4], r2c, scale)) {
            return;
        }
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, FftKernels::dft, e.context(), x, tw, y, stride, nIn, nOut, n, scale, realIn, realOut, weight, total));
    }

    @Override
    public void reference(Node node) {
        FloatArray x = (FloatArray) node.in(0).storage().host();
        FloatArray y = (FloatArray) node.out(0).storage().host();
        int[] a = node.iattr;
        int stride = a[0], nIn = a[1], nOut = a[2], n = a[3], sign = a[4], weight = a[7];
        boolean realIn = a[5] == 1, realOut = a[6] == 1;
        double scale = node.fattr[0];
        int rows = rows(node);
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < nOut; k++) {
                double re = 0, im = 0;
                if (k < n) {
                    for (int j = 0; j < nIn; j++) {
                        double ang = sign * 2 * Math.PI * ((long) k * j % n) / n;
                        double xr = realIn ? x.get(r * stride + j) : x.get((r * stride + j) * 2);
                        double xi = realIn ? 0 : x.get((r * stride + j) * 2 + 1);
                        double w = weight == FftKernels.WEIGHT_IN ? fold(j, n) : 1;
                        re += w * (xr * Math.cos(ang) - xi * Math.sin(ang));
                        im += w * (xr * Math.sin(ang) + xi * Math.cos(ang));
                    }
                    double w = weight == FftKernels.WEIGHT_OUT ? fold(k, n) : 1;
                    re *= scale * w;
                    im *= scale * w;
                }
                int at = r * nOut + k;
                if (realOut) {
                    y.set(at, (float) re);
                } else {
                    y.set(2 * at, (float) re);
                    y.set(2 * at + 1, (float) im);
                }
            }
        }
    }

    private static double fold(int j, int n) {
        return j == 0 || 2 * j == n ? 1 : 2;
    }

    /** The adjoint: conjugate phase, inputs and outputs swapped, weights moved to the other side. */
    @Override
    public Tensor[] backward(Node node, Tensor g) {
        int[] a = node.iattr;
        int weight = a[7] == FftKernels.WEIGHT_IN ? FftKernels.WEIGHT_OUT : a[7] == FftKernels.WEIGHT_OUT ? FftKernels.WEIGHT_IN : FftKernels.WEIGHT_NONE;
        return new Tensor[] { apply(g, Math.min(a[2], a[3]), a[0], a[3], -a[4], node.fattr[0], a[6] == 1, a[5] == 1, weight), null };
    }
}
