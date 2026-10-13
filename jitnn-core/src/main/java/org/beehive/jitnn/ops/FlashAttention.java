package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.FlashKernels128;
import org.beehive.jitnn.kernels.FlashKernels32;
import org.beehive.jitnn.kernels.FlashKernels64;
import org.beehive.jitnn.kernels.FlashTiled64;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Fused scaled dot-product attention ({@link FlashKernels32}, {@link FlashKernels64} and
 * {@link FlashKernels128} by head dimension): softmax(q k^T * scale + mask) v without storing the
 * scores, in one of three layouts that the kernels read and write through strides:
 * <ul>
 * <li>{@link #BHTD}: q [B, H, T, d], k and v [B, H, S, d], o [B, H, T, d] (also [Z, T, d]);</li>
 * <li>{@link #PACKED}: one projection qkv [B, T, 3 * H * d] holding q, k and v side by side
 * (self-attention), o [B, T, H * d]; the gradient is one packed tensor;</li>
 * <li>{@link #BTHD}: q [B, T, H, d], k and v [B, S, H, d], o [B, T, H, d].</li>
 * </ul>
 * The additive mask broadcasts to [B, H, T, S]. Inputs [q, k, v] or [qkv], then the optional
 * mask; outputs [o, lse]. iattr = [Z, T, S, d, causal, heads, mb, mh, mt, ms, layout]; fattr = [scale].
 */
public final class FlashAttention implements Op {

    public static final int BHTD = 0;
    public static final int PACKED = 1;
    public static final int BTHD = 2;
    /** Largest head dimension of the fused kernels. */
    public static final int MAX_HEAD = FlashKernels128.D;
    /** The tiled kernels ({@link FlashTiled64}) for head dimensions 33 to 64 unless {@code -Djitnn.flash=simple}. */
    static final boolean TILED = !"simple".equals(System.getProperty("jitnn.flash"));

    static final FlashAttention OP = new FlashAttention();
    static final Backward BACKWARD = new Backward();

    private FlashAttention() {
    }

    /** True when the fused kernels handle q, k, v in the [B, H, T, d] layout (FP32, v as wide as q). */
    public static boolean supports(Tensor q, Tensor k, Tensor v) {
        return q.dtype() == DType.F32 && k.dtype() == DType.F32 && v.dtype() == DType.F32 && q.rank() >= 3 && q.rank() <= 4 && q.dim(-1) <= MAX_HEAD
                && v.dim(-1) == q.dim(-1) && k.dim(-1) == q.dim(-1) && k.shape().equals(v.shape()) && k.rank() == q.rank();
    }

    /** q [B, H, T, d] (or [Z, T, d]), k and v [.., S, d]: o [B, H, T, d]. */
    public static Tensor apply(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        int t = q.dim(-2);
        int s = k.dim(-2);
        int d = q.dim(-1);
        int heads = q.rank() == 4 ? q.dim(1) : 1;
        return record(new Tensor[] { q, k, v }, q.shape(), q.numel() / (t * d), t, s, d, heads, mask, causal, scale, BHTD);
    }

    /** Self-attention on a packed projection qkv [B, T, 3 * heads * d]: o [B, T, heads * d]. */
    public static Tensor packed(Tensor qkv, int heads, Tensor mask, boolean causal, float scale) {
        Checks.f32(qkv);
        Checks.rank(qkv, 3);
        int b = qkv.dim(0);
        int t = qkv.dim(1);
        int dim = qkv.dim(2) / 3;
        int d = dim / heads;
        if (3 * dim != qkv.dim(2) || d * heads != dim || d > MAX_HEAD) {
            throw new IllegalArgumentException("packed attention: " + qkv.shape() + " with " + heads + " heads");
        }
        return record(new Tensor[] { qkv }, Shape.of(b, t, dim), b * heads, t, t, d, heads, mask, causal, scale, PACKED);
    }

    /** q [B, T, H, d], k and v [B, S, H, d]: o [B, T, H, d]. */
    public static Tensor bthd(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        Checks.f32(q, k, v);
        Checks.rank(q, 4);
        if (!k.shape().equals(v.shape()) || k.rank() != 4 || k.dim(2) != q.dim(2) || k.dim(3) != q.dim(3) || q.dim(3) > MAX_HEAD) {
            throw new IllegalArgumentException("attention [B, T, H, d]: " + q.shape() + ", " + k.shape() + ", " + v.shape());
        }
        int heads = q.dim(2);
        return record(new Tensor[] { q, k, v }, q.shape(), q.dim(0) * heads, q.dim(1), k.dim(1), q.dim(3), heads, mask, causal, scale, BTHD);
    }

    private static Tensor record(Tensor[] qkv, Shape outShape, int z, int t, int s, int d, int heads, Tensor mask, boolean causal, float scale, int layout) {
        int[] strides = { 0, 0, 0, 0 };
        Tensor[] in = qkv;
        if (mask != null) {
            Checks.f32(mask);
            strides = mask.shape().broadcastStrides(Shape.of(z / heads, heads, t, s));
            in = java.util.Arrays.copyOf(qkv, qkv.length + 1);
            in[qkv.length] = mask;
        }
        Tensor o = Tensor.result(outShape, DType.F32);
        Tensor lse = Tensor.result(Shape.of(z * t), DType.F32);
        int[] iattr = { z, t, s, d, causal ? 1 : 0, heads, strides[0], strides[1], strides[2], strides[3], layout };
        Tensor.record(OP, in, new Tensor[] { o, lse }, iattr, new float[] { scale }, true);
        return o;
    }

    /** [heads, the mask strides, then the (offset, batch, head, token) strides of q, k, v and o]. */
    static int[] meta(int[] a) {
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int heads = a[5];
        int dim = heads * d;
        int[] q;
        int[] k;
        int[] v;
        int[] o;
        switch (a[10]) {
            case PACKED -> {
                q = new int[] { 0, t * 3 * dim, d, 3 * dim };
                k = new int[] { dim, t * 3 * dim, d, 3 * dim };
                v = new int[] { 2 * dim, t * 3 * dim, d, 3 * dim };
                o = new int[] { 0, t * dim, d, dim };
            }
            case BTHD -> {
                q = new int[] { 0, t * dim, d, dim };
                k = new int[] { 0, s * dim, d, dim };
                v = k.clone();
                o = q.clone();
            }
            default -> {
                q = new int[] { 0, heads * t * d, t * d, d };
                k = new int[] { 0, heads * s * d, s * d, d };
                v = k.clone();
                o = q.clone();
            }
        }
        int[] m = new int[21];
        m[0] = heads;
        System.arraycopy(a, 6, m, 1, 4);
        System.arraycopy(q, 0, m, 5, 4);
        System.arraycopy(k, 0, m, 9, 4);
        System.arraycopy(v, 0, m, 13, 4);
        System.arraycopy(o, 0, m, 17, 4);
        return m;
    }

    /** Index of row {@code row} of problem z in operand {@code which} (0 q, 1 k, 2 v, 3 o). */
    static int at(int[] meta, int which, int z, int row) {
        int b = z / meta[0];
        int h = z - b * meta[0];
        int m = 5 + 4 * which;
        return meta[m] + b * meta[m + 1] + h * meta[m + 2] + row * meta[m + 3];
    }

    @Override
    public String name() {
        return "flash_attention";
    }

    private static int rowsPerGroup(int d) {
        return d <= FlashKernels32.D ? FlashKernels32.ROWS : (d <= FlashKernels64.D ? FlashKernels64.ROWS : FlashKernels128.ROWS);
    }

    private static WorkerGrid grid(int z, int rows, int d) {
        return Emitter.blocks1D(z * Emitter.ceilDiv(rows, rowsPerGroup(d)), FlashKernels32.THREADS);
    }

    @Override
    public void emit(Emitter e, Node n) {
        int[] a = n.iattr;
        int z = a[0];
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int causal = a[4];
        float scale = n.fattr[0];
        IntArray meta = e.constant(meta(a));
        boolean packed = a[10] == PACKED;
        int qkvCount = packed ? 1 : 3;
        FloatArray q = e.f32(n.in(0));
        FloatArray k = packed ? q : e.f32(n.in(1));
        FloatArray v = packed ? q : e.f32(n.in(2));
        int hasMask = n.inputs.length > qkvCount ? 1 : 0;
        FloatArray mask = hasMask != 0 ? e.f32(n.in(qkvCount)) : e.dummyF32();
        FloatArray o = e.f32(n.out(0));
        if (hasMask == 0 && a[10] == BHTD && e.unread(n.out(1))
                && org.beehive.jitnn.natives.Libs.attention(e, q, k, v, o, z / a[5], a[5], t, s, d, scale, causal != 0)) {
            return;
        }
        FloatArray lse = e.f32(n.out(1));
        WorkerGrid g0 = grid(z, t, d);
        if (d <= FlashKernels32.D) {
            e.task(g0, (g, id) -> g.task(id, FlashKernels32::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else if (d <= FlashKernels64.D && TILED) {
            WorkerGrid tiled = Emitter.blocks2D(z * Emitter.ceilDiv(t, FlashTiled64.BQ), 1, 16, 16);
            e.task(tiled, (g, id) -> g.task(id, FlashTiled64::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else if (d <= FlashKernels64.D) {
            e.task(g0, (g, id) -> g.task(id, FlashKernels64::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else {
            e.task(g0, (g, id) -> g.task(id, FlashKernels128::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        }
    }

    /** Host views of q, k, v and the mask, read through the layout. */
    private record Host(int[] meta, FloatArray q, FloatArray k, FloatArray v, FloatArray mask) {

        static Host of(Node n, int maskAt) {
            boolean packed = n.iattr[10] == PACKED;
            FloatArray q = (FloatArray) n.in(0).storage().host();
            FloatArray k = packed ? q : (FloatArray) n.in(1).storage().host();
            FloatArray v = packed ? q : (FloatArray) n.in(2).storage().host();
            FloatArray mask = maskAt >= 0 ? (FloatArray) n.in(maskAt).storage().host() : null;
            return new Host(FlashAttention.meta(n.iattr), q, k, v, mask);
        }

        double get(int which, int z, int row, int c) {
            FloatArray arr = which == 0 ? q : (which == 1 ? k : v);
            return arr.get(at(meta, which, z, row) + c);
        }

        /** Masked, scaled scores of query row i of problem z in double; NEGATIVE_INFINITY where masked. */
        double[] scores(int[] a, double scale, int z, int i) {
            int s = a[2];
            int d = a[3];
            boolean causal = a[4] != 0;
            double[] row = new double[s];
            for (int j = 0; j < s; j++) {
                if (causal && j > i) {
                    row[j] = Double.NEGATIVE_INFINITY;
                    continue;
                }
                double dot = 0;
                for (int c = 0; c < d; c++) {
                    dot += get(0, z, i, c) * get(1, z, j, c);
                }
                row[j] = dot * scale;
                if (mask != null) {
                    int b = z / a[5];
                    int h = z - b * a[5];
                    row[j] += mask.get(b * a[6] + h * a[7] + i * a[8] + j * a[9]);
                }
            }
            return row;
        }
    }

    /** Softmax of a score row in double. */
    static double[] softmax(double[] row) {
        double m = Double.NEGATIVE_INFINITY;
        for (double x : row) {
            m = Math.max(m, x);
        }
        double sum = 0;
        double[] p = new double[row.length];
        for (int j = 0; j < row.length; j++) {
            p[j] = row[j] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(row[j] - m);
            sum += p[j];
        }
        for (int j = 0; j < row.length; j++) {
            p[j] /= sum;
        }
        return p;
    }

    @Override
    public void reference(Node n) {
        int[] a = n.iattr;
        int z = a[0];
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int qkvCount = a[10] == PACKED ? 1 : 3;
        Host h = Host.of(n, n.inputs.length > qkvCount ? qkvCount : -1);
        FloatArray o = (FloatArray) n.out(0).storage().host();
        FloatArray lse = (FloatArray) n.out(1).storage().host();
        for (int zi = 0; zi < z; zi++) {
            for (int i = 0; i < t; i++) {
                double[] row = h.scores(a, n.fattr[0], zi, i);
                double m = Double.NEGATIVE_INFINITY;
                for (double x : row) {
                    m = Math.max(m, x);
                }
                double sum = 0;
                for (double x : row) {
                    sum += x == Double.NEGATIVE_INFINITY ? 0 : Math.exp(x - m);
                }
                double[] p = softmax(row);
                for (int c = 0; c < d; c++) {
                    double acc = 0;
                    for (int j = 0; j < s; j++) {
                        acc += p[j] * h.get(2, zi, j, c);
                    }
                    o.set(at(h.meta(), 3, zi, i) + c, (float) acc);
                }
                lse.set(zi * t + i, (float) (m + Math.log(sum)));
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        int qkvCount = n.iattr[10] == PACKED ? 1 : 3;
        boolean hasMask = n.inputs.length > qkvCount;
        Tensor[] grads = new Tensor[qkvCount];
        for (int i = 0; i < qkvCount; i++) {
            grads[i] = Tensor.result(n.in(i).shape(), DType.F32);
        }
        Tensor[] in = java.util.Arrays.copyOf(n.inputs, n.inputs.length + 3);
        in[n.inputs.length] = n.out(0);
        in[n.inputs.length + 1] = n.out(1);
        in[n.inputs.length + 2] = gy;
        int[] iattr = java.util.Arrays.copyOf(n.iattr, 12);
        iattr[11] = hasMask ? 1 : 0;
        Tensor.record(BACKWARD, in, grads, iattr, n.fattr.clone(), true);
        return java.util.Arrays.copyOf(grads, n.inputs.length);
    }

    /** Inputs [q, k, v | qkv, (mask), o, lse, dO]; outputs [dq, dk, dv] or [dqkv]. iattr as the forward plus [hasMask]. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "flash_attention_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            int z = a[0];
            int t = a[1];
            int s = a[2];
            int d = a[3];
            int causal = a[4];
            int hasMask = a[11];
            float scale = n.fattr[0];
            IntArray meta = e.constant(meta(a));
            boolean packed = a[10] == PACKED;
            int qkvCount = packed ? 1 : 3;
            int at = qkvCount + hasMask;
            FloatArray q = e.f32(n.in(0));
            FloatArray k = packed ? q : e.f32(n.in(1));
            FloatArray v = packed ? q : e.f32(n.in(2));
            FloatArray mask = hasMask != 0 ? e.f32(n.in(qkvCount)) : e.dummyF32();
            FloatArray o = e.f32(n.in(at));
            FloatArray lse = e.f32(n.in(at + 1));
            FloatArray dout = e.f32(n.in(at + 2));
            FloatArray dq = e.f32(n.out(0));
            FloatArray dk = packed ? dq : e.f32(n.out(1));
            FloatArray dv = packed ? dq : e.f32(n.out(2));
            int rows = z * t;
            FloatArray delta = e.scratch(rows);
            if (d <= FlashKernels64.D && d > FlashKernels32.D && TILED) {
                // backwardQ computes delta for its queries, then backwardKV reads it
                WorkerGrid tq = Emitter.blocks2D(z * Emitter.ceilDiv(t, 32), 1, 16, 16);
                WorkerGrid tk = Emitter.blocks2D(z * Emitter.ceilDiv(s, 32), 1, 16, 16);
                e.task(tq, (g, id) -> g.task(id, FlashTiled64::backwardQ, e.context(), q, k, v, mask, o, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(tk, (g, id) -> g.task(id, FlashTiled64::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
                return;
            }
            e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, FlashKernels64::delta, e.context(), dout, o, delta, meta, t, d, rows));
            WorkerGrid qGrid = grid(z, t, d);
            WorkerGrid kGrid = grid(z, s, d);
            if (d <= FlashKernels32.D) {
                e.task(qGrid, (g, id) -> g.task(id, FlashKernels32::backwardQ, e.context(), q, k, v, mask, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(kGrid, (g, id) -> g.task(id, FlashKernels32::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
            } else if (d <= FlashKernels64.D) {
                e.task(qGrid, (g, id) -> g.task(id, FlashKernels64::backwardQ, e.context(), q, k, v, mask, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(kGrid, (g, id) -> g.task(id, FlashKernels64::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
            } else {
                e.task(qGrid, (g, id) -> g.task(id, FlashKernels128::backwardQ, e.context(), q, k, v, mask, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(kGrid, (g, id) -> g.task(id, FlashKernels128::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
            }
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            int z = a[0];
            int t = a[1];
            int s = a[2];
            int d = a[3];
            double scale = n.fattr[0];
            boolean packed = a[10] == PACKED;
            int qkvCount = packed ? 1 : 3;
            int at = qkvCount + a[11];
            Host h = Host.of(n, a[11] != 0 ? qkvCount : -1);
            int[] meta = h.meta();
            FloatArray dout = (FloatArray) n.in(at + 2).storage().host();
            FloatArray[] out = new FloatArray[3];
            for (int w = 0; w < 3; w++) {
                out[w] = (FloatArray) n.out(packed ? 0 : w).storage().host();
            }
            double[][] acc = { new double[out[0].getSize()], new double[out[1].getSize()], new double[out[2].getSize()] };
            for (int zi = 0; zi < z; zi++) {
                for (int i = 0; i < t; i++) {
                    double[] p = softmax(h.scores(a, scale, zi, i));
                    double[] dp = new double[s];
                    double dot = 0;
                    for (int j = 0; j < s; j++) {
                        for (int c = 0; c < d; c++) {
                            double g = dout.get(at(meta, 3, zi, i) + c);
                            dp[j] += g * h.get(2, zi, j, c);
                            acc[2][at(meta, 2, zi, j) + c] += p[j] * g;
                        }
                        dot += p[j] * dp[j];
                    }
                    for (int j = 0; j < s; j++) {
                        double g = p[j] * (dp[j] - dot) * scale;
                        for (int c = 0; c < d; c++) {
                            acc[0][at(meta, 0, zi, i) + c] += g * h.get(1, zi, j, c);
                            acc[1][at(meta, 1, zi, j) + c] += g * h.get(0, zi, i, c);
                        }
                    }
                }
            }
            if (packed) {
                // q, k and v occupy disjoint columns of the one gradient tensor
                for (int i = 0; i < out[0].getSize(); i++) {
                    out[0].set(i, (float) (acc[0][i] + acc[1][i] + acc[2][i]));
                }
            } else {
                for (int w = 0; w < 3; w++) {
                    for (int i = 0; i < out[w].getSize(); i++) {
                        out[w].set(i, (float) acc[w][i]);
                    }
                }
            }
        }
    }
}
