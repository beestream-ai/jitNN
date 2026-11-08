package ai.beestream.jitnn;

import java.util.Arrays;

/**
 * Linear algebra on matrices and batches of matrices [..., rows, cols], after torch.linalg: LU,
 * triangular and general solves, inverse, determinants, Cholesky, QR, symmetric eigendecomposition
 * and SVD. Every function is differentiable with PyTorch's formulas.
 *
 * <p>The decompositions run one work-group per matrix (see
 * {@link ai.beestream.jitnn.kernels.LinalgKernels}), so they serve batches of small and medium
 * matrices (up to a few hundred rows), the sizes found inside models. Eigenvectors and singular
 * vectors are only defined up to sign; jitNN makes the largest-magnitude component of each
 * eigenvector, and of each left singular vector, positive. General (non-symmetric)
 * eigendecomposition is not provided: it needs complex numbers.
 */
public final class Linalg {

    private Linalg() {
    }

    /** a as [B, rows, cols] and the leading dimensions to restore. */
    private static Tensor batched(Tensor a, String what) {
        if (a.rank() < 2) {
            throw new IllegalArgumentException(what + " needs a matrix, got " + a.shape());
        }
        int b = 1;
        for (int d = 0; d < a.rank() - 2; d++) {
            b *= a.dim(d);
        }
        return a.reshape(b, a.dim(-2), a.dim(-1));
    }

    private static int[] lead(Tensor a, int... tail) {
        int[] dims = Arrays.copyOf(a.shape().dims(), a.rank() - 2 + tail.length);
        System.arraycopy(tail, 0, dims, a.rank() - 2, tail.length);
        return dims;
    }

    /**
     * LU factorisation with partial pivoting, torch.linalg.lu_factor: the packed factors (L unit
     * lower below the diagonal, U on and above) and INT32 row indices perm, with row i of P^T A
     * being row perm[i] of A (0-based, unlike LAPACK's swap list).
     */
    public static Tensor[] luFactor(Tensor a) {
        Tensor[] r = ai.beestream.jitnn.ops.Linalg.lu(batched(a, "luFactor"));
        int n = a.dim(-1);
        return new Tensor[] { r[0].reshape(a.shape().dims()), r[1].reshape(lead(a, n)) };
    }

    /** P, L, U with A = P L U, torch.linalg.lu for square matrices. */
    public static Tensor[] lu(Tensor a) {
        int n = a.dim(-1);
        Tensor[] r = ai.beestream.jitnn.ops.Linalg.lu(batched(a, "lu"));
        Tensor l = r[0].mul(mask(n, true, -1)).add(eye(n));
        Tensor u = r[0].mul(mask(n, false, 0));
        Tensor p = F.oneHot(r[1], n).transpose(-2, -1);
        int[] dims = a.shape().dims();
        return new Tensor[] { p.reshape(dims), l.reshape(dims), u.reshape(dims) };
    }

    /**
     * Solves T X = B for triangular T, torch.linalg.solve_triangular (left side): only T's {@code upper}
     * or lower triangle is read, and with {@code unitDiagonal} not its diagonal either.
     */
    public static Tensor solveTriangular(Tensor t, Tensor b, boolean upper, boolean unitDiagonal) {
        Tensor x = ai.beestream.jitnn.ops.Linalg.triSolve(batched(t, "solveTriangular"), batched(b, "solveTriangular"), upper, unitDiagonal, false);
        return x.reshape(b.shape().dims());
    }

    /** Solves A X = B for square A [..., n, n] and B [..., n, k] through LU with partial pivoting. */
    public static Tensor solve(Tensor a, Tensor b) {
        int n = a.dim(-1);
        Tensor[] f = ai.beestream.jitnn.ops.Linalg.lu(batched(a, "solve"));
        Tensor rhs = F.oneHot(f[1], n).matmul(batched(b, "solve"));
        Tensor y = ai.beestream.jitnn.ops.Linalg.triSolve(f[0], rhs, false, true, false);
        Tensor x = ai.beestream.jitnn.ops.Linalg.triSolve(f[0], y, true, false, false);
        return x.reshape(b.shape().dims());
    }

    /** Inverse of square matrices. */
    public static Tensor inv(Tensor a) {
        Tensor ab = batched(a, "inv");
        int batch = ab.dim(0);
        int n = a.dim(-1);
        float[] id = new float[batch * n * n];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                id[(b * n + i) * n + i] = 1.0f;
            }
        }
        return solve(ab, Tensor.of(id, batch, n, n)).reshape(a.shape().dims());
    }

    /** Diagonals of the packed LU factors [B, n]. */
    private static Tensor luDiagonal(Tensor lu, int n) {
        return lu.mul(eye(n)).sum(-1, false);
    }

    /** Determinants [...] of square matrices. */
    public static Tensor det(Tensor a) {
        int n = a.dim(-1);
        Tensor[] f = ai.beestream.jitnn.ops.Linalg.lu(batched(a, "det"));
        Tensor prod = luDiagonal(f[0], n).cumprod(-1).narrow(-1, n - 1, 1).squeeze(-1);
        return prod.mul(f[2]).reshape(lead(a));
    }

    /** torch.linalg.slogdet: [sign, log |det|], each [...]; only the logarithm carries a gradient. */
    public static Tensor[] slogdet(Tensor a) {
        int n = a.dim(-1);
        Tensor[] f = ai.beestream.jitnn.ops.Linalg.lu(batched(a, "slogdet"));
        Tensor d = luDiagonal(f[0], n);
        Tensor logAbs = d.abs().log().sum(-1, false);
        Tensor sign = d.detach().sign().cumprod(-1).narrow(-1, n - 1, 1).squeeze(-1).mul(f[2]);
        return new Tensor[] { sign.reshape(lead(a)).detach(), logAbs.reshape(lead(a)) };
    }

    /** Lower Cholesky factor L, A = L L^T, of symmetric positive-definite matrices (only the lower triangle is read). */
    public static Tensor cholesky(Tensor a) {
        return ai.beestream.jitnn.ops.Linalg.cholesky(batched(a, "cholesky")).reshape(a.shape().dims());
    }

    /** Reduced QR, torch.linalg.qr: Q [..., m, k] and R [..., k, n], k = min(m, n). */
    public static Tensor[] qr(Tensor a) {
        int m = a.dim(-2);
        int n = a.dim(-1);
        if (m >= n) {
            Tensor p = ai.beestream.jitnn.ops.Linalg.qr(batched(a, "qr"));
            return new Tensor[] { p.narrow(1, 0, m).reshape(lead(a, m, n)), p.narrow(1, m, n).reshape(lead(a, n, n)) };
        }
        // wide: QR of the first m columns, then R's remaining columns are Q^T A
        Tensor ab = batched(a, "qr");
        Tensor[] left = qr(ab.narrow(2, 0, m));
        Tensor rest = left[0].matmul(ab.narrow(2, m, n - m), true, false);
        return new Tensor[] { left[0].reshape(lead(a, m, m)), Tensor.cat(2, left[1], rest).reshape(lead(a, m, n)) };
    }

    /**
     * Eigenvalues (ascending) [..., n] and eigenvectors (columns) [..., n, n] of symmetric matrices,
     * torch.linalg.eigh. The whole matrix is read, so it should be symmetric.
     */
    public static Tensor[] eigh(Tensor a) {
        int n = a.dim(-1);
        Tensor p = ai.beestream.jitnn.ops.Linalg.eigh(batched(a, "eigh"));
        return new Tensor[] { p.narrow(1, 0, 1).reshape(lead(a, n)), p.narrow(1, 1, n).reshape(a.shape().dims()) };
    }

    /** Eigenvalues (ascending) of symmetric matrices, torch.linalg.eigvalsh. */
    public static Tensor eigvalsh(Tensor a) {
        return eigh(a)[0];
    }

    /**
     * Reduced SVD, torch.linalg.svd(full_matrices=False): U [..., m, k], S [..., k] descending and
     * V [..., n, k] (V, not V^T), k = min(m, n).
     */
    public static Tensor[] svd(Tensor a) {
        int m = a.dim(-2);
        int n = a.dim(-1);
        Tensor ab = batched(a, "svd");
        if (m >= n) {
            Tensor p = ai.beestream.jitnn.ops.Linalg.svd(ab, false);
            return new Tensor[] { p.narrow(1, 0, m).reshape(lead(a, m, n)), p.narrow(1, m, 1).reshape(lead(a, n)), p.narrow(1, m + 1, n).reshape(lead(a, n, n)) };
        }
        // wide: A^T = U' S V'^T, so A = V' S U'^T; the signs follow V' (A's U)
        Tensor p = ai.beestream.jitnn.ops.Linalg.svd(ab.transpose(1, 2), true);
        return new Tensor[] { p.narrow(1, n + 1, m).reshape(lead(a, m, m)), p.narrow(1, n, 1).reshape(lead(a, m)), p.narrow(1, 0, n).reshape(lead(a, n, m)) };
    }

    /** Singular values (descending), torch.linalg.svdvals. */
    public static Tensor svdvals(Tensor a) {
        return svd(a)[1];
    }

    private static Tensor mask(int n, boolean lower, int k) {
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i * n + j] = (lower ? j <= i + k : j >= i + k) ? 1.0f : 0.0f;
            }
        }
        return Tensor.of(m, n, n);
    }

    private static Tensor eye(int n) {
        return mask(n, true, 0).mul(mask(n, false, 0));
    }
}
