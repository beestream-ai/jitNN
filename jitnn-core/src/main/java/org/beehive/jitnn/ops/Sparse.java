package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.SparseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Sparse (CSR) times dense, see {@link SparseKernels} and {@link org.beehive.jitnn.SparseMatrix}. */
public final class Sparse {

    static final SpMM SPMM = new SpMM();
    static final Sddmm SDDMM = new Sddmm();

    private Sparse() {
    }

    /**
     * A dense for A in CSR (ptr, col, row per entry, values in CSR order); the transpose's CSR (tPtr,
     * tCol, tRow) and tOrder (the A entry of each A^T entry) serve the gradient.
     */
    public static Tensor spmm(Tensor ptr, Tensor col, Tensor row, Tensor values, Tensor dense, int rows, Tensor tPtr, Tensor tCol, Tensor tRow, Tensor tOrder) {
        Checks.f32(values);
        Checks.f32(dense);
        Checks.rank(dense, 2);
        Tensor out = Tensor.result(Shape.of(rows, dense.dim(1)), DType.F32);
        Tensor.record(SPMM, new Tensor[] { ptr, col, row, values, dense, tPtr, tCol, tRow, tOrder }, new Tensor[] { out }, new int[] { rows }, null, true);
        return out;
    }

    static final class SpMM implements Op {

        @Override
        public String name() {
            return "spmm";
        }

        @Override
        public void emit(Emitter e, Node node) {
            IntArray ptr = e.i32(node.in(0));
            IntArray col = e.i32(node.in(1));
            FloatArray val = e.f32(node.in(3));
            FloatArray dense = e.f32(node.in(4));
            FloatArray out = e.f32(node.out(0));
            int feat = node.in(4).dim(1), total = node.iattr[0] * feat;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, SparseKernels::spmm, e.context(), ptr, col, val, dense, out, feat, total));
        }

        @Override
        public void reference(Node node) {
            IntArray ptr = (IntArray) node.in(0).storage().host();
            IntArray col = (IntArray) node.in(1).storage().host();
            FloatArray val = (FloatArray) node.in(3).storage().host();
            FloatArray dense = (FloatArray) node.in(4).storage().host();
            FloatArray out = (FloatArray) node.out(0).storage().host();
            int rows = node.iattr[0], feat = node.in(4).dim(1);
            for (int r = 0; r < rows; r++) {
                for (int f = 0; f < feat; f++) {
                    double acc = 0;
                    for (int e = ptr.get(r); e < ptr.get(r + 1); e++) {
                        acc += (double) val.get(e) * dense.get(col.get(e) * feat + f);
                    }
                    out.set(r * feat + f, (float) acc);
                }
            }
        }

        /** d values = SDDMM(g, dense); d dense = A^T g, a SpMM over the transposed structure. */
        @Override
        public Tensor[] backward(Node node, Tensor g) {
            Tensor dense = node.in(4);
            Tensor gv = Tensor.result(node.in(3).shape(), DType.F32);
            Tensor.record(SDDMM, new Tensor[] { node.in(2), node.in(1), g, dense }, new Tensor[] { gv }, new int[0], null, true);
            Tensor tValues = node.in(3).indexSelect(0, node.in(8));
            Tensor gd = spmm(node.in(5), node.in(6), node.in(7), tValues, g, dense.dim(0), node.in(0), node.in(1), node.in(2), inverse(node.in(8)));
            return new Tensor[] { null, null, null, gv, gd, null, null, null, null };
        }

        private static Tensor inverse(Tensor perm) {
            IntArray p = (IntArray) perm.storage().host();
            int n = perm.dim(0);
            int[] inv = new int[n];
            for (int i = 0; i < n; i++) {
                inv[p.get(i)] = i;
            }
            return Tensor.of(inv, n);
        }
    }

    /** inputs [row, col, a, b]; out[e] = a[row[e]] . b[col[e]]. */
    static final class Sddmm implements Op {

        @Override
        public String name() {
            return "sddmm";
        }

        @Override
        public void emit(Emitter e, Node node) {
            IntArray row = e.i32(node.in(0));
            IntArray col = e.i32(node.in(1));
            FloatArray a = e.f32(node.in(2));
            FloatArray b = e.f32(node.in(3));
            FloatArray out = e.f32(node.out(0));
            int feat = node.in(2).dim(1), nnz = node.out(0).dim(0);
            e.task(Emitter.grid1D(nnz), (g, id) -> g.task(id, SparseKernels::sddmm, e.context(), row, col, a, b, out, feat, nnz));
        }

        @Override
        public void reference(Node node) {
            IntArray row = (IntArray) node.in(0).storage().host();
            IntArray col = (IntArray) node.in(1).storage().host();
            FloatArray a = (FloatArray) node.in(2).storage().host();
            FloatArray b = (FloatArray) node.in(3).storage().host();
            FloatArray out = (FloatArray) node.out(0).storage().host();
            int feat = node.in(2).dim(1), nnz = node.out(0).dim(0);
            for (int e = 0; e < nnz; e++) {
                double acc = 0;
                for (int f = 0; f < feat; f++) {
                    acc += (double) a.get(row.get(e) * feat + f) * b.get(col.get(e) * feat + f);
                }
                out.set(e, (float) acc);
            }
        }
    }
}
