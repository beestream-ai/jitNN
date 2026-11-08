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
package ai.beestream.jitnn.ops;

import java.util.Arrays;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.LinalgKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Decompositions of batches of matrices [B, rows, cols] (see {@link LinalgKernels}): LU, triangular
 * solve, Cholesky, QR, symmetric eigendecomposition and SVD. Each is differentiable with PyTorch's
 * formulas, written with other operators; solve, inverse and determinants are built on LU in
 * {@link ai.beestream.jitnn.Linalg}. Multi-part results (QR, eigh, SVD) come out packed in one
 * tensor so that one gradient reaches every part.
 */
public final class Linalg {

    static final Lu LU = new Lu();
    static final TriSolve TRI_SOLVE = new TriSolve();
    static final Cholesky CHOLESKY = new Cholesky();
    static final Qr QR = new Qr();
    static final Eigh EIGH = new Eigh();
    static final Svd SVD = new Svd();

    /** Jacobi sweeps at most; float32 converges in well under this for the sizes these kernels serve. */
    static final int SWEEPS = 30;

    private Linalg() {
    }

    static int group(int work) {
        return Math.min(256, Math.max(32, Emitter.ceilDiv(work, 32) * 32));
    }

    private static void square(Tensor a, String what) {
        Checks.f32(a);
        Checks.rank(a, 3);
        if (a.dim(1) != a.dim(2)) {
            throw new IllegalArgumentException(what + " needs square matrices, got " + a.shape());
        }
    }

    // ---------------------------------------------------------------- public entry points (rank 3)

    /** [packed LU [B, n, n], perm INT32 [B, n], sign [B]]. */
    public static Tensor[] lu(Tensor a) {
        square(a, "LU");
        int b = a.dim(0);
        int n = a.dim(1);
        Tensor lu = Tensor.result(a.shape(), DType.F32);
        Tensor perm = Tensor.result(Shape.of(b, n), DType.I32);
        Tensor sign = Tensor.result(Shape.of(b), DType.F32);
        Tensor pivot = Tensor.result(Shape.of(b), DType.I32);
        Tensor.record(LU, new Tensor[] { a }, new Tensor[] { lu, perm, sign, pivot }, new int[0], null, true);
        return new Tensor[] { lu, perm.detach(), sign.detach() };
    }

    /** X with op(T) X = R; op(T) = T or its transpose, lower or upper, unit diagonal or not. */
    public static Tensor triSolve(Tensor t, Tensor r, boolean upper, boolean unit, boolean transpose) {
        square(t, "triangular solve");
        Checks.f32(r);
        Checks.rank(r, 3);
        if (r.dim(0) != t.dim(0) || r.dim(1) != t.dim(1)) {
            throw new IllegalArgumentException("triangular solve: " + t.shape() + " and right-hand side " + r.shape());
        }
        Tensor x = Tensor.result(r.shape(), DType.F32);
        Tensor.record(TRI_SOLVE, new Tensor[] { t, r }, new Tensor[] { x }, new int[] { upper ? 1 : 0, unit ? 1 : 0, transpose ? 1 : 0 }, null, true);
        return x;
    }

    /** Lower Cholesky factor of the lower triangle of a. */
    public static Tensor cholesky(Tensor a) {
        square(a, "Cholesky");
        Tensor l = Tensor.result(a.shape(), DType.F32);
        Tensor.record(CHOLESKY, new Tensor[] { a }, new Tensor[] { l }, new int[0], null, true);
        return l;
    }

    /** Packed reduced QR [B, m + n, n] of a [B, m, n], m >= n. */
    public static Tensor qr(Tensor a) {
        Checks.f32(a);
        Checks.rank(a, 3);
        int b = a.dim(0), m = a.dim(1), n = a.dim(2);
        if (m < n) {
            throw new IllegalArgumentException("QR needs rows >= columns, got " + a.shape());
        }
        Tensor out = Tensor.result(Shape.of(b, m + n, n), DType.F32);
        Tensor r = Tensor.result(a.shape(), DType.F32);
        Tensor v = Tensor.result(a.shape(), DType.F32);
        Tensor.record(QR, new Tensor[] { a }, new Tensor[] { out, r, v }, new int[0], null, true);
        return out;
    }

    /** Packed eigendecomposition [B, n + 1, n] of symmetric a: eigenvalues ascending, then the eigenvectors as columns. */
    public static Tensor eigh(Tensor a) {
        square(a, "eigh");
        int b = a.dim(0), n = a.dim(1);
        Tensor out = Tensor.result(Shape.of(b, n + 1, n), DType.F32);
        Tensor.record(EIGH, new Tensor[] { a }, new Tensor[] { out, Tensor.result(a.shape(), DType.F32), Tensor.result(a.shape(), DType.F32),
                Tensor.result(Shape.of(b, n + 2), DType.F32), Tensor.result(Shape.of(b), DType.I32), Tensor.result(Shape.of(b, n), DType.I32) }, new int[0], null, true);
        return out;
    }

    /** Packed reduced SVD [B, m + 1 + n, n] of a [B, m, n], m >= n: U, singular values descending, V. */
    public static Tensor svd(Tensor a, boolean signOnV) {
        Checks.f32(a);
        Checks.rank(a, 3);
        int b = a.dim(0), m = a.dim(1), n = a.dim(2);
        if (m < n) {
            throw new IllegalArgumentException("SVD kernel needs rows >= columns, got " + a.shape());
        }
        Tensor out = Tensor.result(Shape.of(b, m + 1 + n, n), DType.F32);
        Tensor.record(SVD, new Tensor[] { a }, new Tensor[] { out, Tensor.result(a.shape(), DType.F32), Tensor.result(Shape.of(b, n, n), DType.F32),
                Tensor.result(Shape.of(b, n + 2), DType.F32), Tensor.result(Shape.of(b), DType.I32), Tensor.result(Shape.of(b, n), DType.I32) }, new int[] { signOnV ? 1 : 0 },
                null, true);
        return out;
    }

    // ---------------------------------------------------------------- helpers for the gradients

    static Tensor mT(Tensor x) {
        return x.transpose(-2, -1);
    }

    /** Lower (lower = true, j <= i + k) or upper (j >= i + k) triangle mask [n, n]. */
    static Tensor mask(int n, boolean lower, int k) {
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i * n + j] = (lower ? j <= i + k : j >= i + k) ? 1.0f : 0.0f;
            }
        }
        return Tensor.of(m, n, n);
    }

    static Tensor eye(int n) {
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            m[i * n + i] = 1.0f;
        }
        return Tensor.of(m, n, n);
    }

    /** X with X T = G, T stored lower or upper: T^T X^T = G^T. */
    static Tensor solveRight(Tensor t, Tensor g, boolean upper, boolean unit) {
        return mT(triSolve(t, mT(g), !upper, unit, true));
    }

    // ---------------------------------------------------------------- operators

    /** inputs [A]; outputs [LU, perm, sign, pivot scratch]. */
    static final class Lu implements Op {

        @Override
        public String name() {
            return "lu";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray a = e.f32(node.in(0));
            FloatArray lu = e.f32(node.out(0));
            IntArray perm = e.i32(node.out(1));
            FloatArray sign = e.f32(node.out(2));
            IntArray pivot = e.i32(node.out(3));
            int b = node.in(0).dim(0), n = node.in(0).dim(1), group = group(n * n);
            e.task(Emitter.blocks1D(b, group), (g, id) -> g.task(id, LinalgKernels::lu, e.context(), a, lu, perm, sign, pivot, n, group));
        }

        @Override
        public void reference(Node node) {
            FloatArray a = (FloatArray) node.in(0).storage().host();
            FloatArray lu = (FloatArray) node.out(0).storage().host();
            IntArray perm = (IntArray) node.out(1).storage().host();
            FloatArray sign = (FloatArray) node.out(2).storage().host();
            int batch = node.in(0).dim(0), n = node.in(0).dim(1);
            for (int b = 0; b < batch; b++) {
                double[][] m = load(a, b * n * n, n, n);
                int[] p = new int[n];
                Arrays.setAll(p, i -> i);
                double sg = 1;
                for (int k = 0; k < n; k++) {
                    int piv = k;
                    for (int i = k + 1; i < n; i++) {
                        if (Math.abs((float) m[i][k]) > Math.abs((float) m[piv][k])) {
                            piv = i;
                        }
                    }
                    if (piv != k) {
                        double[] t = m[k];
                        m[k] = m[piv];
                        m[piv] = t;
                        int ti = p[k];
                        p[k] = p[piv];
                        p[piv] = ti;
                        sg = -sg;
                    }
                    if (m[k][k] != 0) {
                        for (int i = k + 1; i < n; i++) {
                            m[i][k] /= m[k][k];
                        }
                    }
                    for (int i = k + 1; i < n; i++) {
                        for (int j = k + 1; j < n; j++) {
                            m[i][j] -= m[i][k] * m[k][j];
                        }
                    }
                }
                store(lu, b * n * n, m);
                for (int i = 0; i < n; i++) {
                    perm.set(b * n + i, p[i]);
                }
                sign.set(b, (float) sg);
            }
        }

        /** PyTorch's linalg_lu_backward for square matrices, from the packed gradient. */
        @Override
        public Tensor[] backward(Node node, Tensor g) {
            Tensor lu = node.out(0);
            int n = lu.dim(1);
            Tensor strict = mask(n, true, -1);
            Tensor upperM = mask(n, false, 0);
            Tensor l = lu.mul(strict).add(eye(n));
            Tensor u = lu.mul(upperM);
            Tensor gl = g.mul(strict);
            Tensor gu = g.mul(upperM);
            Tensor grad = l.matmul(gl, true, false).mul(strict).add(gu.matmul(u, false, true).mul(upperM));
            grad = solveRight(mT(u), grad, false, false);
            grad = triSolve(lu, grad, true, true, true);
            Tensor pt = F.oneHot(node.out(1), n);
            return new Tensor[] { pt.matmul(grad, true, false) };
        }
    }

    /** inputs [T, R]; outputs [X]; iattr = [upper, unit, transpose]. */
    static final class TriSolve implements Op {

        @Override
        public String name() {
            return "tri_solve";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray t = e.f32(node.in(0));
            FloatArray r = e.f32(node.in(1));
            FloatArray x = e.f32(node.out(0));
            int n = node.in(1).dim(1), k = node.in(1).dim(2), total = node.in(1).dim(0) * k;
            int upper = node.iattr[0], unit = node.iattr[1], trans = node.iattr[2];
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, LinalgKernels::triSolve, e.context(), t, r, x, n, k, total, upper, unit, trans));
        }

        @Override
        public void reference(Node node) {
            FloatArray t = (FloatArray) node.in(0).storage().host();
            FloatArray r = (FloatArray) node.in(1).storage().host();
            FloatArray x = (FloatArray) node.out(0).storage().host();
            int batch = node.in(1).dim(0), n = node.in(1).dim(1), k = node.in(1).dim(2);
            boolean upper = node.iattr[0] == 1, unit = node.iattr[1] == 1, trans = node.iattr[2] == 1;
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < k; c++) {
                    double[] xs = new double[n];
                    for (int step = 0; step < n; step++) {
                        int i = upper ? n - 1 - step : step;
                        double s = r.get((b * n + i) * k + c);
                        for (int j = upper ? i + 1 : 0; j < (upper ? n : i); j++) {
                            s -= t.get(b * n * n + (trans ? j * n + i : i * n + j)) * xs[j];
                        }
                        if (!unit) {
                            s /= t.get(b * n * n + i * n + i);
                        }
                        xs[i] = s;
                        x.set((b * n + i) * k + c, (float) s);
                    }
                }
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor g) {
            boolean upper = node.iattr[0] == 1, unit = node.iattr[1] == 1, trans = node.iattr[2] == 1;
            Tensor t = node.in(0);
            int n = t.dim(1);
            Tensor gr = triSolve(t, g, !upper, unit, !trans);
            Tensor gop = gr.matmul(node.out(0), false, true).neg().mul(mask(n, !upper, unit ? (upper ? 1 : -1) : 0));
            return new Tensor[] { trans ? mT(gop) : gop, gr };
        }
    }

    /** inputs [A]; outputs [L]. */
    static final class Cholesky implements Op {

        @Override
        public String name() {
            return "cholesky";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray a = e.f32(node.in(0));
            FloatArray l = e.f32(node.out(0));
            int b = node.in(0).dim(0), n = node.in(0).dim(1), group = group(n * n);
            e.task(Emitter.blocks1D(b, group), (g, id) -> g.task(id, LinalgKernels::cholesky, e.context(), a, l, n, group));
        }

        @Override
        public void reference(Node node) {
            FloatArray a = (FloatArray) node.in(0).storage().host();
            FloatArray l = (FloatArray) node.out(0).storage().host();
            int batch = node.in(0).dim(0), n = node.in(0).dim(1);
            for (int b = 0; b < batch; b++) {
                double[][] m = load(a, b * n * n, n, n);
                double[][] out = new double[n][n];
                for (int j = 0; j < n; j++) {
                    double d = m[j][j];
                    for (int k = 0; k < j; k++) {
                        d -= out[j][k] * out[j][k];
                    }
                    out[j][j] = Math.sqrt(d);
                    for (int i = j + 1; i < n; i++) {
                        double s = m[i][j];
                        for (int k = 0; k < j; k++) {
                            s -= out[i][k] * out[j][k];
                        }
                        out[i][j] = s / out[j][j];
                    }
                }
                store(l, b * n * n, out);
            }
        }

        /** PyTorch's cholesky_backward. */
        @Override
        public Tensor[] backward(Node node, Tensor g) {
            Tensor l = node.out(0);
            int n = l.dim(1);
            Tensor ga = l.matmul(g, true, false).mul(mask(n, true, 0));
            ga = ga.add(mT(ga.mul(mask(n, true, -1)))).mul(0.5f);
            ga = triSolve(l, ga, true, false, true);
            ga = solveRight(l, ga, false, false);
            return new Tensor[] { ga };
        }
    }

    /** inputs [A]; outputs [packed Q R, r scratch, v scratch]. */
    static final class Qr implements Op {

        @Override
        public String name() {
            return "qr";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray a = e.f32(node.in(0));
            FloatArray out = e.f32(node.out(0));
            FloatArray r = e.f32(node.out(1));
            FloatArray v = e.f32(node.out(2));
            int b = node.in(0).dim(0), m = node.in(0).dim(1), n = node.in(0).dim(2), group = group(n);
            e.task(Emitter.blocks1D(b, group), (g, id) -> g.task(id, LinalgKernels::qr, e.context(), a, out, r, v, m, n, group));
        }

        @Override
        public void reference(Node node) {
            FloatArray a = (FloatArray) node.in(0).storage().host();
            FloatArray out = (FloatArray) node.out(0).storage().host();
            int batch = node.in(0).dim(0), m = node.in(0).dim(1), n = node.in(0).dim(2);
            for (int b = 0; b < batch; b++) {
                double[][] r = load(a, b * m * n, m, n);
                double[][] vs = new double[n][m];
                for (int k = 0; k < n; k++) {
                    double sig = 0;
                    for (int i = k + 1; i < m; i++) {
                        sig += r[i][k] * r[i][k];
                    }
                    if (sig == 0) {
                        continue;
                    }
                    double x0 = r[k][k];
                    double norm = Math.sqrt(x0 * x0 + sig);
                    double alpha = x0 >= 0 ? -norm : norm;
                    double vn = Math.sqrt((x0 - alpha) * (x0 - alpha) + sig);
                    vs[k][k] = (x0 - alpha) / vn;
                    for (int i = k + 1; i < m; i++) {
                        vs[k][i] = r[i][k] / vn;
                    }
                    reflect(vs[k], r, k, m, n);
                }
                double[][] q = new double[m][n];
                for (int i = 0; i < n; i++) {
                    q[i][i] = 1;
                }
                for (int k = n - 1; k >= 0; k--) {
                    reflect(vs[k], q, k, m, n);
                }
                int ob = b * (m + n) * n;
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        out.set(ob + i * n + j, (float) q[i][j]);
                    }
                }
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        out.set(ob + (m + i) * n + j, j >= i ? (float) r[i][j] : 0.0f);
                    }
                }
            }
        }

        private static void reflect(double[] v, double[][] x, int k, int m, int n) {
            for (int j = 0; j < n; j++) {
                double dot = 0;
                for (int i = k; i < m; i++) {
                    dot += v[i] * x[i][j];
                }
                for (int i = k; i < m; i++) {
                    x[i][j] -= 2 * v[i] * dot;
                }
            }
        }

        /** PyTorch's linalg_qr_backward for m >= n. */
        @Override
        public Tensor[] backward(Node node, Tensor g) {
            int m = node.in(0).dim(1), n = node.in(0).dim(2);
            Tensor q = node.out(0).narrow(1, 0, m);
            Tensor r = node.out(0).narrow(1, m, n);
            Tensor gq = g.narrow(1, 0, m);
            Tensor gr = g.narrow(1, m, n);
            Tensor mm = gr.matmul(r, false, true).sub(q.matmul(gq, true, false)).mul(mask(n, false, 0));
            Tensor sym = mm.add(mT(mm.mul(mask(n, false, 1))));
            Tensor ga = q.matmul(sym).add(gq);
            return new Tensor[] { solveRight(mT(r), ga, false, false) };
        }
    }

    /** inputs [A]; outputs [packed values and vectors, w, v, cs, flag, perm (scratch)]. */
    static final class Eigh implements Op {

        @Override
        public String name() {
            return "eigh";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray a = e.f32(node.in(0));
            FloatArray out = e.f32(node.out(0));
            FloatArray w = e.f32(node.out(1));
            FloatArray v = e.f32(node.out(2));
            FloatArray cs = e.f32(node.out(3));
            IntArray flag = e.i32(node.out(4));
            IntArray perm = e.i32(node.out(5));
            int b = node.in(0).dim(0), n = node.in(0).dim(1), group = group(n * n / 2);
            e.task(Emitter.blocks1D(b, group), (g, id) -> g.task(id, LinalgKernels::eigh, e.context(), a, out, w, v, cs, flag, perm, n, SWEEPS, group));
        }

        @Override
        public void reference(Node node) {
            FloatArray a = (FloatArray) node.in(0).storage().host();
            FloatArray out = (FloatArray) node.out(0).storage().host();
            int batch = node.in(0).dim(0), n = node.in(0).dim(1);
            for (int b = 0; b < batch; b++) {
                double[][] w = load(a, b * n * n, n, n);
                double[][] v = new double[n][n];
                for (int i = 0; i < n; i++) {
                    v[i][i] = 1;
                }
                for (int sweep = 0; sweep < 60; sweep++) {
                    boolean rotated = false;
                    for (int p = 0; p < n; p++) {
                        for (int q = p + 1; q < n; q++) {
                            if (Math.abs(w[p][q]) <= 1e-15 * Math.sqrt(Math.abs(w[p][p] * w[q][q])) || Math.abs(w[p][q]) < 1e-300) {
                                continue;
                            }
                            rotated = true;
                            double theta = (w[q][q] - w[p][p]) / (2 * w[p][q]);
                            double t = Math.copySign(1 / (Math.abs(theta) + Math.sqrt(theta * theta + 1)), theta);
                            double c = 1 / Math.sqrt(t * t + 1), s = t * c;
                            for (int j = 0; j < n; j++) {
                                double wp = w[p][j], wq = w[q][j];
                                w[p][j] = c * wp - s * wq;
                                w[q][j] = s * wp + c * wq;
                            }
                            for (int j = 0; j < n; j++) {
                                double wp = w[j][p], wq = w[j][q];
                                w[j][p] = c * wp - s * wq;
                                w[j][q] = s * wp + c * wq;
                                double vp = v[j][p], vq = v[j][q];
                                v[j][p] = c * vp - s * vq;
                                v[j][q] = s * vp + c * vq;
                            }
                        }
                    }
                    if (!rotated) {
                        break;
                    }
                }
                Integer[] order = new Integer[n];
                Arrays.setAll(order, i -> i);
                Arrays.sort(order, (x, y) -> Double.compare(w[x][x], w[y][y]));
                int ob = b * (n + 1) * n;
                for (int i = 0; i < n; i++) {
                    int col = order[i];
                    out.set(ob + i, (float) w[col][col]);
                    double sgn = largestSign(v, col, n);
                    for (int r = 0; r < n; r++) {
                        out.set(ob + (1 + r) * n + i, (float) (sgn * v[r][col]));
                    }
                }
            }
        }

        /** PyTorch's linalg_eig_backward for Hermitian matrices. */
        @Override
        public Tensor[] backward(Node node, Tensor g) {
            int n = node.in(0).dim(1);
            Tensor vals = node.out(0).narrow(1, 0, 1);
            Tensor v = node.out(0).narrow(1, 1, n);
            Tensor gl = g.narrow(1, 0, 1);
            Tensor gv = g.narrow(1, 1, n);
            Tensor k = v.matmul(gv, true, false);
            k = k.sub(mT(k)).mul(0.5f);
            Tensor eye = eye(n);
            Tensor e = vals.sub(mT(vals)).add(eye);
            Tensor inner = k.div(e).mul(eye.neg().add(1.0f)).add(eye.mul(gl));
            return new Tensor[] { v.matmul(inner).matmul(v, false, true) };
        }
    }

    /** inputs [A]; outputs [packed U, S, V; u, v, cs, flag, perm (scratch)]; iattr = [signOnV]. */
    static final class Svd implements Op {

        @Override
        public String name() {
            return "svd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray a = e.f32(node.in(0));
            FloatArray out = e.f32(node.out(0));
            FloatArray u = e.f32(node.out(1));
            FloatArray v = e.f32(node.out(2));
            FloatArray cs = e.f32(node.out(3));
            IntArray flag = e.i32(node.out(4));
            IntArray perm = e.i32(node.out(5));
            int b = node.in(0).dim(0), m = node.in(0).dim(1), n = node.in(0).dim(2), signOnV = node.iattr[0], group = group((n + 1) / 2 * (m + n));
            e.task(Emitter.blocks1D(b, group), (g, id) -> g.task(id, LinalgKernels::svd, e.context(), a, out, u, v, cs, flag, perm, m, n, SWEEPS, signOnV, group));
        }

        @Override
        public void reference(Node node) {
            FloatArray a = (FloatArray) node.in(0).storage().host();
            FloatArray out = (FloatArray) node.out(0).storage().host();
            int batch = node.in(0).dim(0), m = node.in(0).dim(1), n = node.in(0).dim(2);
            boolean signOnV = node.iattr[0] == 1;
            for (int b = 0; b < batch; b++) {
                double[][] u = load(a, b * m * n, m, n);
                double[][] v = new double[n][n];
                for (int i = 0; i < n; i++) {
                    v[i][i] = 1;
                }
                for (int sweep = 0; sweep < 60; sweep++) {
                    boolean rotated = false;
                    for (int p = 0; p < n; p++) {
                        for (int q = p + 1; q < n; q++) {
                            double al = 0, be = 0, ga = 0;
                            for (int r = 0; r < m; r++) {
                                al += u[r][p] * u[r][p];
                                be += u[r][q] * u[r][q];
                                ga += u[r][p] * u[r][q];
                            }
                            if (Math.abs(ga) <= 1e-15 * Math.sqrt(al * be) || Math.abs(ga) < 1e-300) {
                                continue;
                            }
                            rotated = true;
                            double zeta = (be - al) / (2 * ga);
                            double t = Math.copySign(1 / (Math.abs(zeta) + Math.sqrt(zeta * zeta + 1)), zeta);
                            double c = 1 / Math.sqrt(t * t + 1), s = t * c;
                            for (int r = 0; r < m; r++) {
                                double xp = u[r][p], xq = u[r][q];
                                u[r][p] = c * xp - s * xq;
                                u[r][q] = s * xp + c * xq;
                            }
                            for (int r = 0; r < n; r++) {
                                double xp = v[r][p], xq = v[r][q];
                                v[r][p] = c * xp - s * xq;
                                v[r][q] = s * xp + c * xq;
                            }
                        }
                    }
                    if (!rotated) {
                        break;
                    }
                }
                double[] sv = new double[n];
                for (int j = 0; j < n; j++) {
                    double s = 0;
                    for (int r = 0; r < m; r++) {
                        s += u[r][j] * u[r][j];
                    }
                    sv[j] = Math.sqrt(s);
                }
                Integer[] order = new Integer[n];
                Arrays.setAll(order, i -> i);
                Arrays.sort(order, (x, y) -> Double.compare(sv[y], sv[x]));
                int ob = b * (m + 1 + n) * n;
                for (int i = 0; i < n; i++) {
                    int col = order[i];
                    double inv = sv[col] > 0 ? 1 / sv[col] : 0;
                    double sgn = signOnV ? largestSign(v, col, n) : largestSign(u, col, m);
                    for (int r = 0; r < m; r++) {
                        out.set(ob + r * n + i, (float) (sgn * inv * u[r][col]));
                    }
                    out.set(ob + m * n + i, (float) sv[col]);
                    for (int r = 0; r < n; r++) {
                        out.set(ob + (m + 1 + r) * n + i, (float) (sgn * v[r][col]));
                    }
                }
            }
        }

        /** PyTorch's svd_backward for m >= n (real). */
        @Override
        public Tensor[] backward(Node node, Tensor g) {
            int m = node.in(0).dim(1), n = node.in(0).dim(2);
            Tensor p = node.out(0);
            Tensor u = p.narrow(1, 0, m);
            Tensor s = p.narrow(1, m, 1);
            Tensor v = p.narrow(1, m + 1, n);
            Tensor gu = g.narrow(1, 0, m);
            Tensor gs = g.narrow(1, m, 1);
            Tensor gv = g.narrow(1, m + 1, n);
            Tensor uhgu = u.matmul(gu, true, false);
            uhgu = uhgu.sub(mT(uhgu));
            Tensor vhgv = v.matmul(gv, true, false);
            vhgv = vhgv.sub(mT(vhgv));
            Tensor eye = eye(n);
            Tensor s2 = s.square();
            Tensor e = s2.sub(mT(s2)).add(eye);
            Tensor ret = uhgu.mul(s).add(mT(s).mul(vhgv)).div(e).add(eye.mul(gs));
            Tensor ga = u.matmul(ret);
            if (m > n) {
                Tensor gusinv = gu.div(s);
                ga = ga.add(gusinv).sub(u.matmul(u.matmul(gusinv, true, false)));
            }
            return new Tensor[] { ga.matmul(v, false, true) };
        }
    }

    // ---------------------------------------------------------------- host helpers

    static double[][] load(FloatArray a, int base, int rows, int cols) {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = a.get(base + i * cols + j);
            }
        }
        return m;
    }

    static void store(FloatArray a, int base, double[][] m) {
        int cols = m[0].length;
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < cols; j++) {
                a.set(base + i * cols + j, (float) m[i][j]);
            }
        }
    }

    /** The sign of column {@code col}'s largest-magnitude component (the first on ties). */
    static double largestSign(double[][] x, int col, int rows) {
        double big = 0;
        double sgn = 1;
        for (int r = 0; r < rows; r++) {
            if (Math.abs(x[r][col]) > big) {
                big = Math.abs(x[r][col]);
                sgn = x[r][col] < 0 ? -1 : 1;
            }
        }
        return sgn;
    }
}
