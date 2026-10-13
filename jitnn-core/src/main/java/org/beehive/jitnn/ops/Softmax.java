package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.SoftmaxKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Softmax and log-softmax along the last axis. iattr = [rows, cols, logMode]. */
public final class Softmax implements Op {

    static final Softmax OP = new Softmax();
    static final Backward BACKWARD = new Backward();

    private Softmax() {
    }

    public static Tensor apply(Tensor x, boolean log) {
        Checks.f32(x);
        if (x.rank() == 0) {
            throw new IllegalArgumentException("softmax needs at least one dimension");
        }
        int cols = x.dim(-1);
        int rows = x.numel() / Math.max(cols, 1);
        Tensor y = Tensor.result(x.shape(), DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { y }, new int[] { rows, cols, log ? 1 : 0 }, null, true);
        return y;
    }

    @Override
    public String name() {
        return "softmax";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray y = e.f32(n.out(0));
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        int log = n.iattr[2];
        if (log == 0 && org.beehive.jitnn.natives.Libs.softmax(e, x, y, rows, cols)) {
            return;
        }
        if (cols <= SoftmaxKernels.CACHED_COLS) {
            e.task(Emitter.groups(rows), (g, id) -> g.task(id, SoftmaxKernels::softmaxRowsCached, e.context(), x, y, cols, log));
        } else {
            e.task(Emitter.groups(rows), (g, id) -> g.task(id, SoftmaxKernels::softmaxRows, e.context(), x, y, cols, log));
        }
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray y = (FloatArray) n.out(0).storage().host();
        int rows = n.iattr[0];
        int cols = n.iattr[1];
        boolean log = n.iattr[2] != 0;
        for (int r = 0; r < rows; r++) {
            double m = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                m = Math.max(m, x.get(r * cols + c));
            }
            double s = 0;
            for (int c = 0; c < cols; c++) {
                s += Math.exp(x.get(r * cols + c) - m);
            }
            for (int c = 0; c < cols; c++) {
                double v = x.get(r * cols + c) - m;
                y.set(r * cols + c, (float) (log ? v - Math.log(s) : Math.exp(v) / s));
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor dx = Tensor.result(n.in(0).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { n.out(0), gy }, new Tensor[] { dx }, n.iattr.clone(), null, true);
        return new Tensor[] { dx };
    }

    /** dx from the forward output and dy. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "softmax_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray y = e.f32(n.in(0));
            FloatArray dy = e.f32(n.in(1));
            FloatArray dx = e.f32(n.out(0));
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            int log = n.iattr[2];
            e.task(Emitter.groups(rows), (g, id) -> g.task(id, SoftmaxKernels::softmaxBackwardRows, e.context(), y, dy, dx, cols, log));
        }

        @Override
        public void reference(Node n) {
            FloatArray y = (FloatArray) n.in(0).storage().host();
            FloatArray dy = (FloatArray) n.in(1).storage().host();
            FloatArray dx = (FloatArray) n.out(0).storage().host();
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            boolean log = n.iattr[2] != 0;
            for (int r = 0; r < rows; r++) {
                double dot = 0;
                for (int c = 0; c < cols; c++) {
                    int i = r * cols + c;
                    dot += log ? dy.get(i) : (double) dy.get(i) * y.get(i);
                }
                for (int c = 0; c < cols; c++) {
                    int i = r * cols + c;
                    dx.set(i, (float) (log ? dy.get(i) - Math.exp(y.get(i)) * dot : y.get(i) * (dy.get(i) - dot)));
                }
            }
        }
    }
}
