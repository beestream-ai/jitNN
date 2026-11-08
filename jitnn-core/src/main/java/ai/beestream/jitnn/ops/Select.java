package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.ElementwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Selection by a mask ({@code where}) and one-hot encoding. */
public final class Select {

    static final Where WHERE = new Where();
    static final OneHot ONE_HOT = new OneHot();

    private Select() {
    }

    /** cond != 0 ? a : b, the three broadcast together (PyTorch's torch.where with a float mask). */
    public static Tensor where(Tensor cond, Tensor a, Tensor b) {
        Checks.f32(cond, a, b);
        Shape out = Shape.broadcast(Shape.broadcast(cond.shape(), a.shape()), b.shape());
        Tensor y = Tensor.result(out, DType.F32);
        Tensor.record(WHERE, new Tensor[] { cond.expand(out), a.expand(out), b.expand(out) }, new Tensor[] { y }, null, null, true);
        return y;
    }

    /** INT32 labels [...] to [..., classes] of 0 and 1; labels outside [0, classes) give a zero row. */
    public static Tensor oneHot(Tensor labels, int classes) {
        if (labels.dtype() != DType.I32) {
            throw new IllegalArgumentException("one-hot needs INT32 labels, got " + labels.dtype());
        }
        int[] dims = java.util.Arrays.copyOf(labels.shape().dims(), labels.rank() + 1);
        dims[labels.rank()] = classes;
        Tensor y = Tensor.result(Shape.of(dims), DType.F32);
        Tensor.record(ONE_HOT, new Tensor[] { labels }, new Tensor[] { y }, new int[] { classes }, null, true);
        return y;
    }

    static final class Where implements Op {

        @Override
        public String name() {
            return "where";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray c = e.f32(n.in(0));
            FloatArray a = e.f32(n.in(1));
            FloatArray b = e.f32(n.in(2));
            FloatArray y = e.f32(n.out(0));
            int count = n.out(0).numel();
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::where, e.context(), c, a, b, y, count));
        }

        @Override
        public void reference(Node n) {
            FloatArray c = (FloatArray) n.in(0).storage().host();
            FloatArray a = (FloatArray) n.in(1).storage().host();
            FloatArray b = (FloatArray) n.in(2).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                y.set(i, c.get(i) != 0 ? a.get(i) : b.get(i));
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor c = n.in(0);
            Tensor zero = Tensor.scalar(0.0f);
            Tensor ga = n.in(1).requiresGrad() ? where(c, gy, zero) : null;
            Tensor gb = n.in(2).requiresGrad() ? where(c, zero, gy) : null;
            return new Tensor[] { null, ga, gb };
        }
    }

    static final class OneHot implements Op {

        @Override
        public String name() {
            return "one_hot";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray labels = e.i32(n.in(0));
            FloatArray y = e.f32(n.out(0));
            int classes = n.iattr[0];
            int count = n.out(0).numel();
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::oneHot, e.context(), labels, y, classes, count));
        }

        @Override
        public void reference(Node n) {
            IntArray labels = (IntArray) n.in(0).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            int classes = n.iattr[0];
            for (int i = 0; i < n.out(0).numel(); i++) {
                y.set(i, labels.get(i / classes) == i % classes ? 1 : 0);
            }
        }
    }
}
