/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn;

import java.util.Arrays;

/**
 * The structure of a sparse matrix (which entries are stored), built once on the host from
 * coordinate lists; the values are a separate tensor, in the order the coordinates were given,
 * so they can be learned. The structure is kept on the device in CSR form and, for the gradient
 * with respect to the dense operand, in CSR form of the transpose.
 */
public final class SparseMatrix {

    final int rows;
    final int cols;
    final int nnz;
    /** CSR of A: row pointers, column of each stored entry, row of each stored entry, and its coordinate-list index. */
    final Tensor ptr;
    final Tensor col;
    final Tensor row;
    final Tensor order;
    /** CSR of A^T, and for each of its entries the matching entry of A's CSR. */
    final Tensor tPtr;
    final Tensor tCol;
    final Tensor tRow;
    final Tensor tOrder;

    private SparseMatrix(int rows, int cols, int[] r, int[] c) {
        this.rows = rows;
        this.cols = cols;
        this.nnz = r.length;
        int[][] a = csr(rows, r, c, identity(nnz));
        int[] csrRow = a[2];
        int[] csrCol = a[1];
        int[][] t = csr(cols, csrCol, csrRow, identity(nnz));
        ptr = Tensor.of(a[0], rows + 1);
        col = Tensor.of(a[1], nnz);
        row = Tensor.of(a[2], nnz);
        order = Tensor.of(a[3], nnz);
        tPtr = Tensor.of(t[0], cols + 1);
        tCol = Tensor.of(t[1], nnz);
        tRow = Tensor.of(t[2], nnz);
        tOrder = Tensor.of(t[3], nnz);
    }

    private static int[] identity(int n) {
        int[] p = new int[n];
        Arrays.setAll(p, i -> i);
        return p;
    }

    /** [pointers, columns, rows, source index] of the entries sorted by row (stable). */
    private static int[][] csr(int rows, int[] r, int[] c, int[] src) {
        int n = r.length;
        int[] ptr = new int[rows + 1];
        for (int e = 0; e < n; e++) {
            if (r[e] < 0 || r[e] >= rows) {
                throw new IllegalArgumentException("sparse entry " + e + " has row " + r[e] + " outside [0, " + rows + ")");
            }
            ptr[r[e] + 1]++;
        }
        for (int i = 0; i < rows; i++) {
            ptr[i + 1] += ptr[i];
        }
        int[] next = Arrays.copyOf(ptr, rows);
        int[] cols = new int[n];
        int[] rs = new int[n];
        int[] from = new int[n];
        for (int e = 0; e < n; e++) {
            int at = next[r[e]]++;
            cols[at] = c[e];
            rs[at] = r[e];
            from[at] = src[e];
        }
        return new int[][] { ptr, cols, rs, from };
    }

    /** A rows x cols structure with entries at (rowIndex[e], colIndex[e]); duplicates add up. */
    public static SparseMatrix fromCoo(int rows, int cols, int[] rowIndex, int[] colIndex) {
        if (rowIndex.length != colIndex.length) {
            throw new IllegalArgumentException("sparse: " + rowIndex.length + " row indices but " + colIndex.length + " column indices");
        }
        for (int v : colIndex) {
            if (v < 0 || v >= cols) {
                throw new IllegalArgumentException("sparse column " + v + " outside [0, " + cols + ")");
            }
        }
        return new SparseMatrix(rows, cols, rowIndex.clone(), colIndex.clone());
    }

    /** The number of rows. */
    public int rows() {
        return rows;
    }

    /** The number of columns. */
    public int cols() {
        return cols;
    }

    /** The number of stored entries. */
    public int nnz() {
        return nnz;
    }

    /** A dense with A's stored values {@code values} [nnz] (coordinate order); dense [cols, feat] gives [rows, feat]. */
    public Tensor mm(Tensor values, Tensor dense) {
        if (values.rank() != 1 || values.dim(0) != nnz) {
            throw new IllegalArgumentException("sparse values must be [" + nnz + "], got " + values.shape());
        }
        return ai.beestream.jitnn.ops.Sparse.spmm(this.ptr, col, row, values.indexSelect(0, order), dense, rows, tPtr, tCol, tRow, tOrder);
    }

    /**
     * The normalised adjacency of a graph convolutional network (Kipf and Welling, 2017), D^-1/2 (A +
     * I) D^-1/2 for the directed edges src[e] -> dst[e] of an n-node graph (messages flow from source
     * to destination, as in PyTorch Geometric's GCNConv).
     */
    public static Valued gcn(int n, int[] src, int[] dst) {
        int e = src.length;
        int[] r = new int[e + n];
        int[] c = new int[e + n];
        float[] deg = new float[n];
        for (int i = 0; i < e; i++) {
            r[i] = dst[i];
            c[i] = src[i];
            deg[dst[i]] += 1;
        }
        for (int i = 0; i < n; i++) {
            r[e + i] = i;
            c[e + i] = i;
            deg[i] += 1;
        }
        float[] v = new float[e + n];
        for (int i = 0; i < e + n; i++) {
            v[i] = (float) (1.0 / Math.sqrt((double) deg[r[i]] * deg[c[i]]));
        }
        return new Valued(fromCoo(n, n, r, c), Tensor.of(v, e + n));
    }

    /** A structure with fixed values. */
    public record Valued(SparseMatrix matrix, Tensor values) {

        /** This matrix times dense [cols, feat]. */
        public Tensor mm(Tensor dense) {
            return matrix.mm(values, dense);
        }
    }
}
