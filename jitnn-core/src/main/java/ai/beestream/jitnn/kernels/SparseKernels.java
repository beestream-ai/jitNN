package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Sparse matrices in CSR form (row pointers, column indices, values) times dense matrices. */
public final class SparseKernels {

    private SparseKernels() {
    }

    /** out [rows, feat] = A dense, dense [cols, feat]; one thread per output element, adjacent threads on adjacent features. */
    public static void spmm(KernelContext kc, IntArray ptr, IntArray col, FloatArray val, FloatArray dense, FloatArray out, int feat, int total) {
        int id = kc.globalIdx;
        if (id < total) {
            int r = id / feat;
            int f = id % feat;
            float acc = 0.0f;
            for (int e = ptr.get(r); e < ptr.get(r + 1); e++) {
                acc += val.get(e) * dense.get(col.get(e) * feat + f);
            }
            out.set(id, acc);
        }
    }

    /** out[e] = a[row[e], :] . b[col[e], :], the sampled dense-dense product (the gradient of the values). */
    public static void sddmm(KernelContext kc, IntArray row, IntArray col, FloatArray a, FloatArray b, FloatArray out, int feat, int nnz) {
        int e = kc.globalIdx;
        if (e < nnz) {
            int ra = row.get(e) * feat;
            int rb = col.get(e) * feat;
            float acc = 0.0f;
            for (int f = 0; f < feat; f++) {
                acc += a.get(ra + f) * b.get(rb + f);
            }
            out.set(e, acc);
        }
    }
}
