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

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Fused scaled dot-product attention ({@link FlashKernels32}, {@link FlashKernels64} and
 * {@link FlashKernels128} by head dimension): softmax(q k^T * scale + mask) v
 * without storing the scores. q [B, H, T, d] (or [Z, T, d]), k and v [..., S, d] with the same
 * leading dimensions; the optional additive mask broadcasts to [B, H, T, S]. Inputs [q, k, v] or
 * [q, k, v, mask]; outputs [o, lse]. iattr = [Z, T, S, d, causal, heads, mb, mh, mt, ms]; fattr = [scale].
 */
public final class FlashAttention implements Op {

    static final FlashAttention OP = new FlashAttention();
    static final Backward BACKWARD = new Backward();

    private FlashAttention() {
    }

    /** True when the fused kernels handle these operands (FP32, head dimension at most 128, v as wide as q). */
    public static boolean supports(Tensor q, Tensor k, Tensor v) {
        return q.dtype() == DType.F32 && k.dtype() == DType.F32 && v.dtype() == DType.F32 && q.rank() >= 3 && q.rank() <= 4 && q.dim(-1) <= FlashKernels128.D
                && v.dim(-1) == q.dim(-1) && k.dim(-1) == q.dim(-1) && k.shape().equals(v.shape()) && k.rank() == q.rank();
    }

    public static Tensor apply(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        int t = q.dim(-2);
        int s = k.dim(-2);
        int d = q.dim(-1);
        int z = q.numel() / (t * d);
        int heads = q.rank() == 4 ? q.dim(1) : 1;
        int batch = z / heads;
        int[] strides = { 0, 0, 0, 0 };
        if (mask != null) {
            Checks.f32(mask);
            strides = mask.shape().broadcastStrides(Shape.of(batch, heads, t, s));
        }
        Tensor o = Tensor.result(q.shape(), DType.F32);
        Tensor lse = Tensor.result(Shape.of(z * t), DType.F32);
        Tensor[] in = mask != null ? new Tensor[] { q, k, v, mask } : new Tensor[] { q, k, v };
        int[] iattr = { z, t, s, d, causal ? 1 : 0, heads, strides[0], strides[1], strides[2], strides[3] };
        Tensor.record(OP, in, new Tensor[] { o, lse }, iattr, new float[] { scale }, true);
        return o;
    }

    @Override
    public String name() {
        return "flash_attention";
    }

    private static int groups(int z, int rows, int block) {
        return z * Emitter.ceilDiv(rows, block);
    }

    @Override
    public void emit(Emitter e, Node n) {
        int[] a = n.iattr;
        int z = a[0];
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int causal = a[4];
        int heads = a[5];
        int mb = a[6];
        int mh = a[7];
        int mt = a[8];
        int ms = a[9];
        float scale = n.fattr[0];
        uk.ac.manchester.tornado.api.types.arrays.IntArray meta = e.constant(heads, mb, mh, mt, ms);
        FloatArray q = e.f32(n.in(0));
        FloatArray k = e.f32(n.in(1));
        FloatArray v = e.f32(n.in(2));
        int hasMask = n.inputs.length > 3 ? 1 : 0;
        FloatArray mask = hasMask != 0 ? e.f32(n.in(3)) : e.dummyF32();
        FloatArray o = e.f32(n.out(0));
        FloatArray lse = e.f32(n.out(1));
        int rows = d <= FlashKernels32.D ? FlashKernels32.ROWS : (d <= FlashKernels64.D ? FlashKernels64.ROWS : FlashKernels128.ROWS);
        WorkerGrid grid = Emitter.blocks1D(groups(z, t, rows), FlashKernels32.THREADS);
        if (d <= FlashKernels32.D) {
            e.task(grid, (g, id) -> g.task(id, FlashKernels32::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else if (d <= FlashKernels64.D) {
            e.task(grid, (g, id) -> g.task(id, FlashKernels64::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else {
            e.task(grid, (g, id) -> g.task(id, FlashKernels128::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        }
    }

    /** Scores, masked and scaled, of one (z, i) row in double; NEGATIVE_INFINITY where masked. */
    static double[] scores(Node n, int zi, int i) {
        int[] a = n.iattr;
        int t = a[1];
        int s = a[2];
        int d = a[3];
        boolean causal = a[4] != 0;
        int heads = a[5];
        double scale = n.fattr[0];
        FloatArray q = (FloatArray) n.in(0).storage().host();
        FloatArray k = (FloatArray) n.in(1).storage().host();
        boolean hasMask = n.op == OP ? n.inputs.length > 3 : a[10] != 0;
        FloatArray mask = hasMask ? (FloatArray) n.in(3).storage().host() : null;
        double[] row = new double[s];
        for (int j = 0; j < s; j++) {
            if (causal && j > i) {
                row[j] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double dot = 0;
            for (int c = 0; c < d; c++) {
                dot += (double) q.get((zi * t + i) * d + c) * k.get((zi * s + j) * d + c);
            }
            row[j] = dot * scale;
            if (mask != null) {
                int b = zi / heads;
                int h = zi - b * heads;
                row[j] += mask.get(b * a[6] + h * a[7] + i * a[8] + j * a[9]);
            }
        }
        return row;
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
        FloatArray v = (FloatArray) n.in(2).storage().host();
        FloatArray o = (FloatArray) n.out(0).storage().host();
        FloatArray lse = (FloatArray) n.out(1).storage().host();
        for (int zi = 0; zi < z; zi++) {
            for (int i = 0; i < t; i++) {
                double[] row = scores(n, zi, i);
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
                        acc += p[j] * v.get((zi * s + j) * d + c);
                    }
                    o.set((zi * t + i) * d + c, (float) acc);
                }
                lse.set(zi * t + i, (float) (m + Math.log(sum)));
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor q = n.in(0);
        Tensor k = n.in(1);
        Tensor v = n.in(2);
        boolean hasMask = n.inputs.length > 3;
        Tensor dq = Tensor.result(q.shape(), DType.F32);
        Tensor dk = Tensor.result(k.shape(), DType.F32);
        Tensor dv = Tensor.result(v.shape(), DType.F32);
        Tensor[] in = hasMask ? new Tensor[] { q, k, v, n.in(3), n.out(0), n.out(1), gy } : new Tensor[] { q, k, v, n.out(0), n.out(1), gy };
        int[] iattr = java.util.Arrays.copyOf(n.iattr, 11);
        iattr[10] = hasMask ? 1 : 0;
        Tensor.record(BACKWARD, in, new Tensor[] { dq, dk, dv }, iattr, n.fattr.clone(), true);
        return hasMask ? new Tensor[] { dq, dk, dv, null } : new Tensor[] { dq, dk, dv };
    }

    /** Inputs [q, k, v, (mask), o, lse, dO]; outputs [dq, dk, dv]. iattr as the forward plus [hasMask]. */
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
            int heads = a[5];
            int mb = a[6];
            int mh = a[7];
            int mt = a[8];
            int ms = a[9];
            int hasMask = a[10];
            float scale = n.fattr[0];
            uk.ac.manchester.tornado.api.types.arrays.IntArray meta = e.constant(heads, mb, mh, mt, ms);
            int at = hasMask != 0 ? 4 : 3;
            FloatArray q = e.f32(n.in(0));
            FloatArray k = e.f32(n.in(1));
            FloatArray v = e.f32(n.in(2));
            FloatArray mask = hasMask != 0 ? e.f32(n.in(3)) : e.dummyF32();
            FloatArray o = e.f32(n.in(at));
            FloatArray lse = e.f32(n.in(at + 1));
            FloatArray dout = e.f32(n.in(at + 2));
            FloatArray dq = e.f32(n.out(0));
            FloatArray dk = e.f32(n.out(1));
            FloatArray dv = e.f32(n.out(2));
            int rows = z * t;
            FloatArray delta = e.scratch(rows);
            e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, FlashKernels64::delta, e.context(), dout, o, delta, d, rows));
            int rowsPer = d <= FlashKernels32.D ? FlashKernels32.ROWS : (d <= FlashKernels64.D ? FlashKernels64.ROWS : FlashKernels128.ROWS);
            WorkerGrid qGrid = Emitter.blocks1D(groups(z, t, rowsPer), FlashKernels32.THREADS);
            WorkerGrid kGrid = Emitter.blocks1D(groups(z, s, rowsPer), FlashKernels32.THREADS);
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
            int at = a[10] != 0 ? 4 : 3;
            FloatArray q = (FloatArray) n.in(0).storage().host();
            FloatArray k = (FloatArray) n.in(1).storage().host();
            FloatArray v = (FloatArray) n.in(2).storage().host();
            FloatArray dout = (FloatArray) n.in(at + 2).storage().host();
            double[] dq = new double[z * t * d];
            double[] dk = new double[z * s * d];
            double[] dv = new double[z * s * d];
            for (int zi = 0; zi < z; zi++) {
                for (int i = 0; i < t; i++) {
                    double[] p = softmax(scores(n, zi, i));
                    double[] dp = new double[s];
                    double dot = 0;
                    for (int j = 0; j < s; j++) {
                        for (int c = 0; c < d; c++) {
                            dp[j] += (double) dout.get((zi * t + i) * d + c) * v.get((zi * s + j) * d + c);
                            dv[(zi * s + j) * d + c] += p[j] * dout.get((zi * t + i) * d + c);
                        }
                        dot += p[j] * dp[j];
                    }
                    for (int j = 0; j < s; j++) {
                        double g = p[j] * (dp[j] - dot) * scale;
                        for (int c = 0; c < d; c++) {
                            dq[(zi * t + i) * d + c] += g * k.get((zi * s + j) * d + c);
                            dk[(zi * s + j) * d + c] += g * q.get((zi * t + i) * d + c);
                        }
                    }
                }
            }
            write((FloatArray) n.out(0).storage().host(), dq);
            write((FloatArray) n.out(1).storage().host(), dk);
            write((FloatArray) n.out(2).storage().host(), dv);
        }

        private static void write(FloatArray out, double[] values) {
            for (int i = 0; i < values.length; i++) {
                out.set(i, (float) values[i]);
            }
        }
    }
}
