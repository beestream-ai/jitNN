package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.ElementwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Embedding lookup: out[..., d] = weight[index[...], d]. The backward pass is a deterministic
 * segment sum: the indices are grouped by row on the host (offsets and positions, like CSR), and
 * each thread of the gradient kernel sums one (row, column) over its positions in order. No
 * atomics, so gradients are bit-reproducible.
 */
public final class Embedding implements Op {

    static final Embedding OP = new Embedding();
    static final Backward BACKWARD = new Backward();

    private Embedding() {
    }

    /**
     * Rows of {@code weight} [V, D] selected by the INT32 {@code index}; the result has shape
     * index.shape + [D]. Rows equal to {@code paddingIdx} (or none when negative) get no gradient.
     */
    public static Tensor apply(Tensor index, Tensor weight, int paddingIdx) {
        if (index.dtype() != DType.I32) {
            throw new IllegalArgumentException("embedding indices must be INT32, got " + index);
        }
        Checks.f32(weight);
        Checks.rank(weight, 2);
        int[] idims = index.shape().dims();
        int[] dims = java.util.Arrays.copyOf(idims, idims.length + 1);
        dims[idims.length] = weight.dim(1);
        Tensor out = Tensor.result(Shape.of(dims), DType.F32);
        Tensor.record(OP, new Tensor[] { index, weight }, new Tensor[] { out }, new int[] { index.numel(), weight.dim(0), weight.dim(1), paddingIdx }, null, true);
        return out;
    }

    @Override
    public String name() {
        return "embedding";
    }

    @Override
    public void emit(Emitter e, Node n) {
        IntArray index = e.i32(n.in(0));
        FloatArray weight = e.f32(n.in(1));
        FloatArray out = e.f32(n.out(0));
        int dim = n.iattr[2];
        int count = n.iattr[0] * dim;
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::gatherRows, e.context(), weight, index, out, dim, count));
    }

    @Override
    public void reference(Node n) {
        IntArray index = (IntArray) n.in(0).storage().host();
        FloatArray weight = (FloatArray) n.in(1).storage().host();
        FloatArray out = (FloatArray) n.out(0).storage().host();
        int vocab = n.iattr[1];
        int dim = n.iattr[2];
        for (int t = 0; t < n.iattr[0]; t++) {
            int row = index.get(t);
            if (row < 0 || row >= vocab) {
                throw new IndexOutOfBoundsException("embedding index " + row + " for " + vocab + " rows");
            }
            for (int d = 0; d < dim; d++) {
                out.set(t * dim + d, weight.get(row * dim + d));
            }
        }
    }

    @Override
    public boolean cpu(Node n) {
        return CpuKernels.embedding(n.in(0), n.in(1), n.out(0), n.iattr[0], n.iattr[1], n.iattr[2]);
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        if (!n.in(1).requiresGrad()) {
            return new Tensor[] { null, null };
        }
        int tokens = n.iattr[0];
        int vocab = n.iattr[1];
        int padding = n.iattr[3];
        // The indices normally come from the host (a data loader); otherwise this reads them back.
        int[] idx = n.in(0).storage().isHostValid() ? hostInts(n.in(0)) : n.in(0).toIntArray();
        int[] offsets = new int[vocab + 1];
        for (int v : idx) {
            if (v < 0 || v >= vocab) {
                throw new IndexOutOfBoundsException("embedding index " + v + " for " + vocab + " rows");
            }
            if (v != padding) {
                offsets[v + 1]++;
            }
        }
        for (int v = 0; v < vocab; v++) {
            offsets[v + 1] += offsets[v];
        }
        int[] fill = java.util.Arrays.copyOf(offsets, vocab);
        int[] positions = new int[Math.max(offsets[vocab], 1)];
        for (int t = 0; t < tokens; t++) {
            int v = idx[t];
            if (v != padding) {
                positions[fill[v]++] = t;
            }
        }
        Tensor dw = Tensor.result(n.in(1).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { gy, Tensor.of(offsets, vocab + 1), Tensor.of(positions, positions.length) }, new Tensor[] { dw }, n.iattr.clone(), null, true);
        return new Tensor[] { null, dw };
    }

    private static int[] hostInts(Tensor t) {
        IntArray a = (IntArray) t.storage().host();
        int[] out = new int[t.numel()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i);
        }
        return out;
    }

    /** dW from dy, the row offsets [V + 1] and the token positions grouped by row. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "embedding_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray dy = e.f32(n.in(0));
            IntArray offsets = e.i32(n.in(1));
            IntArray positions = e.i32(n.in(2));
            FloatArray dw = e.f32(n.out(0));
            int dim = n.iattr[2];
            int count = n.iattr[1] * dim;
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::segmentSumRows, e.context(), dy, offsets, positions, dw, dim, count));
        }

        @Override
        public boolean cpu(Node n) {
            return CpuKernels.embeddingBackward(n.in(0), n.in(1), n.in(2), n.out(0), n.iattr[1], n.iattr[2]);
        }

        @Override
        public void reference(Node n) {
            FloatArray dy = (FloatArray) n.in(0).storage().host();
            IntArray offsets = (IntArray) n.in(1).storage().host();
            IntArray positions = (IntArray) n.in(2).storage().host();
            FloatArray dw = (FloatArray) n.out(0).storage().host();
            int vocab = n.iattr[1];
            int dim = n.iattr[2];
            for (int v = 0; v < vocab; v++) {
                for (int d = 0; d < dim; d++) {
                    double acc = 0;
                    for (int k = offsets.get(v); k < offsets.get(v + 1); k++) {
                        acc += dy.get(positions.get(k) * dim + d);
                    }
                    dw.set(v * dim + d, (float) acc);
                }
            }
        }
    }
}
