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

/** Reorders dimensions into a new contiguous tensor. iattr = the permutation. */
public final class Permute implements Op {

    static final Permute OP = new Permute();

    private Permute() {
    }

    public static Tensor apply(Tensor x, int... axes) {
        Checks.f32(x);
        int[] order = axes.clone();
        int r = x.rank();
        if (order.length != r) {
            throw new IllegalArgumentException("permutation of length " + order.length + " for rank " + r);
        }
        boolean[] seen = new boolean[r];
        int[] dims = new int[r];
        boolean identity = true;
        for (int i = 0; i < r; i++) {
            int a = x.shape().axis(order[i]);
            if (seen[a]) {
                throw new IllegalArgumentException("repeated axis in permutation");
            }
            seen[a] = true;
            dims[i] = x.dim(a);
            identity &= a == i;
            order[i] = a;
        }
        if (identity) {
            return x;
        }
        Tensor out = Tensor.result(Shape.of(dims), DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, order.clone(), null, true);
        return out;
    }

    static int[] meta(Shape in, int[] order) {
        int r = order.length;
        int[] strides = in.strides();
        int[] dims = in.dims();
        int[] meta = new int[1 + 2 * r];
        meta[0] = r;
        for (int d = 0; d < r; d++) {
            meta[1 + d] = dims[order[d]];
            meta[1 + r + d] = strides[order[d]];
        }
        return meta;
    }

    @Override
    public String name() {
        return "permute";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray out = e.f32(n.out(0));
        IntArray meta = e.constant(meta(n.in(0).shape(), n.iattr));
        int count = n.out(0).numel();
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::permute, e.context(), x, out, meta, count));
    }

    @Override
    public void reference(Node n) {
        int[] meta = meta(n.in(0).shape(), n.iattr);
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray o = (FloatArray) n.out(0).storage().host();
        int r = meta[0];
        for (int i = 0; i < n.out(0).numel(); i++) {
            int rem = i;
            int src = 0;
            for (int d = r - 1; d >= 0; d--) {
                int c = rem % meta[1 + d];
                rem /= meta[1 + d];
                src += c * meta[1 + r + d];
            }
            o.set(i, x.get(src));
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        int[] inverse = new int[n.iattr.length];
        for (int i = 0; i < inverse.length; i++) {
            inverse[n.iattr[i]] = i;
        }
        return new Tensor[] { gy.permute(inverse) };
    }
}
