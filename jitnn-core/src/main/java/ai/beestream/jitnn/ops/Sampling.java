package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.RandomKernels;
import ai.beestream.jitnn.kernels.SampleKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Top-k along the last axis, and multinomial sampling of one index per row. */
public final class Sampling {

    static final TopK TOPK = new TopK();
    static final TopKGrad TOPK_GRAD = new TopKGrad();
    static final Multinomial MULTINOMIAL = new Multinomial();

    private Sampling() {
    }

    /**
     * The k largest values along the last axis and their INT32 indices, in descending order (ties:
     * lower index first): [values, indices], each of shape [..., k].
     */
    public static Tensor[] topk(Tensor x, int k) {
        Checks.f32(x);
        int cols = x.dim(-1);
        if (k < 1 || k > cols) {
            throw new IllegalArgumentException("top-" + k + " of " + cols + " entries");
        }
        int rows = x.numel() / cols;
        int[] dims = x.shape().dims();
        dims[dims.length - 1] = k;
        Tensor values = Tensor.result(Shape.of(dims), DType.F32);
        Tensor indices = Tensor.result(Shape.of(dims), DType.I32);
        Tensor.record(TOPK, new Tensor[] { x }, new Tensor[] { values, indices }, new int[] { rows, cols, k }, null, true);
        return new Tensor[] { values, indices };
    }

    /**
     * One index per row of non-negative, unnormalised weights [..., V], drawn with probability
     * proportional to the weight. {@code seed} is an INT32 [2] host tensor (a new seed per call
     * draws new samples without recompiling) and {@code stream} separates independent draws.
     */
    public static Tensor multinomial(Tensor weights, Tensor seed, int stream) {
        Checks.f32(weights);
        if (seed.dtype() != DType.I32 || seed.numel() != 2) {
            throw new IllegalArgumentException("the seed must be an INT32 tensor of 2 elements");
        }
        int cols = weights.dim(-1);
        int rows = weights.numel() / cols;
        int[] dims = java.util.Arrays.copyOf(weights.shape().dims(), weights.rank() - 1);
        Tensor out = Tensor.result(Shape.of(dims), DType.I32);
        Tensor.record(MULTINOMIAL, new Tensor[] { weights, seed }, new Tensor[] { out }, new int[] { rows, cols, stream }, null, true);
        return out;
    }

    /** iattr = [rows, cols, k]. */
    static final class TopK implements Op {

        @Override
        public String name() {
            return "topk";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            FloatArray values = e.f32(n.out(0));
            IntArray indices = e.i32(n.out(1));
            int cols = n.iattr[1];
            int k = n.iattr[2];
            e.task(Emitter.groups(n.iattr[0]), (g, id) -> g.task(id, SampleKernels::topkRows, e.context(), x, values, indices, cols, k));
        }

        @Override
        public void reference(Node n) {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            FloatArray values = (FloatArray) n.out(0).storage().host();
            IntArray indices = (IntArray) n.out(1).storage().host();
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            int k = n.iattr[2];
            for (int r = 0; r < rows; r++) {
                Integer[] order = new Integer[cols];
                for (int c = 0; c < cols; c++) {
                    order[c] = c;
                }
                int base = r * cols;
                java.util.Arrays.sort(order, (a, b) -> {
                    int cmp = Float.compare(x.get(base + b), x.get(base + a));
                    return cmp != 0 ? cmp : Integer.compare(a, b);
                });
                for (int q = 0; q < k; q++) {
                    values.set(r * k + q, x.get(base + order[q]));
                    indices.set(r * k + q, order[q]);
                }
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor dx = Tensor.result(n.in(0).shape(), DType.F32);
            Tensor.record(TOPK_GRAD, new Tensor[] { gy, n.out(1) }, new Tensor[] { dx }, n.iattr.clone(), null, true);
            return new Tensor[] { dx };
        }
    }

    /** dx from dvalues and the picked indices. iattr = [rows, cols, k]. */
    static final class TopKGrad implements Op {

        @Override
        public String name() {
            return "topk_grad";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray dy = e.f32(n.in(0));
            IntArray indices = e.i32(n.in(1));
            FloatArray dx = e.f32(n.out(0));
            int cols = n.iattr[1];
            int k = n.iattr[2];
            int count = n.iattr[0] * cols;
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, SampleKernels::topkBackward, e.context(), dy, indices, dx, cols, k, count));
        }

        @Override
        public void reference(Node n) {
            FloatArray dy = (FloatArray) n.in(0).storage().host();
            IntArray indices = (IntArray) n.in(1).storage().host();
            FloatArray dx = (FloatArray) n.out(0).storage().host();
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            int k = n.iattr[2];
            for (int i = 0; i < rows * cols; i++) {
                dx.set(i, 0.0f);
            }
            for (int r = 0; r < rows; r++) {
                for (int q = 0; q < k; q++) {
                    int at = r * cols + indices.get(r * k + q);
                    dx.set(at, dx.get(at) + dy.get(r * k + q));
                }
            }
        }
    }

    /** iattr = [rows, cols, stream]. */
    static final class Multinomial implements Op {

        @Override
        public String name() {
            return "multinomial";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray w = e.f32(n.in(0));
            IntArray seed = e.i32(n.in(1));
            IntArray out = e.i32(n.out(0));
            int cols = n.iattr[1];
            int stream = n.iattr[2];
            e.task(Emitter.groups(n.iattr[0]), (g, id) -> g.task(id, SampleKernels::multinomialRows, e.context(), w, seed, out, cols, stream, RandomKernels.ROUNDS));
        }

        @Override
        public void reference(Node n) {
            FloatArray w = (FloatArray) n.in(0).storage().host();
            IntArray seed = (IntArray) n.in(1).storage().host();
            IntArray out = (IntArray) n.out(0).storage().host();
            int rows = n.iattr[0];
            int cols = n.iattr[1];
            int stream = n.iattr[2];
            for (int r = 0; r < rows; r++) {
                double total = 0;
                for (int c = 0; c < cols; c++) {
                    total += w.get(r * cols + c);
                }
                double u = RandomFill.unit(RandomFill.philox(r, stream, 0, 0, seed.get(0), seed.get(1))[0]) * total;
                double acc = 0;
                int pick = -1;
                int last = 0;
                for (int c = 0; c < cols; c++) {
                    double v = w.get(r * cols + c);
                    if (v > 0) {
                        last = c;
                        acc += v;
                        if (pick < 0 && acc > u) {
                            pick = c;
                        }
                    }
                }
                out.set(r, pick < 0 ? last : pick);
            }
        }
    }
}
