package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.ElementwiseKernels;
import org.beehive.jitnn.kernels.Int8Kernels;
import org.beehive.jitnn.kernels.MmaInt8Kernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * INT8 products, as ONNX's quantised models specify them: activations quantised with a per-tensor
 * scale and zero point (ONNX QuantizeLinear, round half to even, saturating), weights as INT8 with
 * a per-output scale and a zero point of 0, and the products accumulated exactly in INT32. UINT8
 * activations are stored as INT8 minus 128, so both operands fit the tensor cores' signed
 * products; the result is dequantised to FP32 in the same kernel (see {@link MmaInt8Kernels}).
 * Operands are packed four values to an INT32 element: activations [M, Kp / 4] row by row, weights
 * in the kernel's tile order, with K padded with zeros to Kp, a multiple of 32.
 */
public final class Int8 {

    static final Quantize QUANTIZE = new Quantize();
    static final Linear LINEAR = new Linear();
    private static final int STEP = MmaInt8Kernels.STEP;
    private static final int TILE = MmaInt8Kernels.TILE;
    /** Work-groups the K splits aim for when the output alone gives fewer. */
    private static final int TARGET_GROUPS = 96;

    private Int8() {
    }

    /** K padded to a multiple of 32. */
    static int padded(int k) {
        return (k + STEP - 1) / STEP * STEP;
    }

    /**
     * INT8 weights prepared for {@link #linear}: B [N, K] packed in tile order, the per-output
     * scale [N] and the sum of each output's weights [N] (INT32).
     */
    public record Weights(Tensor packed, Tensor scale, Tensor rowSums, int n, int k) {

        /**
         * From the integer values of an ONNX weight {@code q} [K, N] (row-major, as MatMul's second
         * operand) and its scale, one value or one per column.
         */
        public static Weights ofColumns(long[] q, int k, int n, float[] scale) {
            int kp = padded(k);
            int steps = kp / STEP;
            int blocks = (n + TILE - 1) / TILE;
            IntArray packed = new IntArray(Math.max(blocks * steps * TILE * STEP / 4, 1));
            IntArray sums = new IntArray(Math.max(n, 1));
            FloatArray s = new FloatArray(Math.max(n, 1));
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int i = 0; i < k; i++) {
                    long v = q[i * n + j];
                    if (v < -128 || v > 127) {
                        throw new IllegalArgumentException("INT8 weight value " + v + " out of range");
                    }
                    sum += (int) v;
                }
                sums.set(j, sum);
                s.set(j, scale[scale.length == 1 ? 0 : j]);
            }
            // block (nb, step), panel p of 8 columns, row kr of k pairs, column pair jp:
            // bytes B(k, j), B(k + 1, j), B(k, j + 1), B(k + 1, j + 1) for k = step * 32 + 2 kr, j = nb * 128 + p * 8 + 2 jp
            for (int nb = 0; nb < blocks; nb++) {
                for (int st = 0; st < steps; st++) {
                    for (int e = 0; e < TILE * STEP / 4; e++) {
                        int j = nb * TILE + (e >> 6) * 8 + ((e & 3) << 1);
                        int kk = st * STEP + (((e >> 2) & 15) << 1);
                        int v = value(q, k, n, kk, j) | value(q, k, n, kk + 1, j) << 8 | value(q, k, n, kk, j + 1) << 16 | value(q, k, n, kk + 1, j + 1) << 24;
                        packed.set((nb * steps + st) * TILE * STEP / 4 + e, v);
                    }
                }
            }
            return new Weights(Tensor.persistent(packed, DType.I32, packed.getSize()), Tensor.persistent(s, DType.F32, Math.max(n, 1)),
                    Tensor.persistent(sums, DType.I32, Math.max(n, 1)), n, k);
        }

        /** Weight (k, j) as a byte, 0 in the padding. */
        private static int value(long[] q, int k, int n, int i, int j) {
            return i < k && j < n ? (int) q[i * n + j] & 0xFF : 0;
        }

        /** Weight (j, i) of B [N, K] (output j, input i) from the packed form. */
        int get(IntArray packed, int j, int i) {
            int steps = padded(k) / STEP;
            int nb = j / TILE, jj = j % TILE, st = i / STEP, kk = i % STEP;
            int e = (jj / 8) * 64 + (kk / 2) * 4 + (jj % 8) / 2;
            int shift = ((jj & 1) << 4) + ((kk & 1) << 3);
            return (byte) (packed.get((nb * steps + st) * TILE * STEP / 4 + e) >> shift);
        }
    }

    /**
     * x [M, K] (FP32) quantised to INT8, packed four to an INT32 element ([M, Kp / 4]):
     * clamp(round_half_even(x / scale) + zp, lo, hi), stored minus 128 for {@code unsigned} (UINT8,
     * 0 .. 255) and as it is otherwise (INT8, -128 .. 127). scale and zp are one-element FP32
     * tensors (constants, or computed, as by DynamicQuantizeLinear).
     */
    public static Tensor quantize(Tensor x, Tensor scale, Tensor zp, boolean unsigned) {
        Checks.f32(x, scale, zp);
        Checks.rank(x, 2);
        int m = x.dim(0), k = x.dim(1);
        Tensor q = Tensor.result(Shape.of(m, padded(k) / 4), DType.I32);
        Tensor.record(QUANTIZE, new Tensor[] { x, scale.reshape(1), zp.reshape(1) }, new Tensor[] { q }, new int[] { unsigned ? 1 : 0, m, k }, null, true);
        return q;
    }

    /**
     * y [M, N] = scale * w.scale[n] * sum_k (q[m, k] - (zp - offset)) * w[n, k] (+ bias[n]): the FP32
     * result of an INT8 product, for q from {@link #quantize} with the same scale and zp and
     * {@code unsigned} (offset 128). bias may be null.
     */
    public static Tensor linear(Tensor q, Tensor scale, Tensor zp, boolean unsigned, Weights w, Tensor bias) {
        int kp = padded(w.k());
        if (q.dtype() != DType.I32 || q.rank() != 2 || q.dim(1) != kp / 4) {
            throw new IllegalArgumentException("INT8 linear: packed input " + q.shape() + " " + q.dtype() + " for K " + w.k());
        }
        int m = q.dim(0), n = w.n();
        int groups = Math.max(1, (m + TILE - 1) / TILE * ((n + TILE - 1) / TILE));
        // K splits when the output tiles alone leave the GPU idle, each split at least 4 steps
        int splits = Math.max(1, Math.min((TARGET_GROUPS + groups - 1) / groups, kp / STEP / 4));
        Tensor y = Tensor.result(Shape.of(m, n), DType.F32);
        Tensor[] in = bias == null ? new Tensor[] { q, scale.reshape(1), zp.reshape(1), w.packed(), w.scale(), w.rowSums() }
                : new Tensor[] { q, scale.reshape(1), zp.reshape(1), w.packed(), w.scale(), w.rowSums(), bias };
        Tensor.record(LINEAR, in, new Tensor[] { y }, new int[] { m, n, w.k(), unsigned ? 128 : 0, bias == null ? 0 : 1, splits }, null, true);
        return y;
    }

    /** INT8 value i of a packed row (sign-extended). */
    private static int unpack(IntArray a, int base, int i) {
        return (byte) (a.get(base + (i >> 2)) >> ((i & 3) << 3));
    }

    /** iattr = [unsigned, M, K]. */
    static final class Quantize implements Op {

        @Override
        public String name() {
            return "quantize_i8";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            FloatArray scale = e.f32(n.in(1));
            FloatArray zp = e.f32(n.in(2));
            IntArray q = e.i32(n.out(0));
            boolean unsigned = n.iattr[0] != 0;
            int m = n.iattr[1], k = n.iattr[2], kp = padded(k);
            float lo = unsigned ? 0 : -128;
            float hi = unsigned ? 255 : 127;
            int offset = unsigned ? 128 : 0;
            int count = m * kp / 4;
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, MmaInt8Kernels::quantize, e.context(), x, q, scale, zp, lo, hi, offset, m, k, kp));
        }

        @Override
        public void reference(Node n) {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            float scale = ((FloatArray) n.in(1).storage().host()).get(0);
            float zp = ((FloatArray) n.in(2).storage().host()).get(0);
            IntArray q = (IntArray) n.out(0).storage().host();
            boolean unsigned = n.iattr[0] != 0;
            int m = n.iattr[1], k = n.iattr[2], lda = padded(k) / 4;
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < lda; c++) {
                    int packed = 0;
                    for (int j = 0; j < 4; j++) {
                        int i = c * 4 + j;
                        int v = 0;
                        if (i < k) {
                            float f = ElementwiseKernels.roundHalfEven(x.get(r * k + i) / scale) + zp;
                            v = (int) Math.max(unsigned ? 0 : -128, Math.min(unsigned ? 255 : 127, f)) - (unsigned ? 128 : 0);
                        }
                        packed |= (v & 0xFF) << (j * 8);
                    }
                    q.set(r * lda + c, packed);
                }
            }
        }
    }

    /** iattr = [M, N, K, offset, hasBias, splits]. */
    static final class Linear implements Op {

        @Override
        public String name() {
            return "linear_i8";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int m = n.iattr[0], cols = n.iattr[1], k = n.iattr[2], offset = n.iattr[3], hasBias = n.iattr[4], splits = n.iattr[5];
            int kp = padded(k);
            IntArray q = e.i32(n.in(0));
            FloatArray sx = e.f32(n.in(1));
            FloatArray zp = e.f32(n.in(2));
            IntArray w = e.i32(n.in(3));
            FloatArray sw = e.f32(n.in(4));
            IntArray sums = e.i32(n.in(5));
            FloatArray bias = hasBias != 0 ? e.f32(n.in(6)) : e.dummyF32();
            FloatArray y = e.f32(n.out(0));
            if (Engine.onCuda() && MatMul.tensorCores()) {
                IntArray partial = splits > 1 ? e.scratchI32(splits * m * cols) : e.constant(new int[] { 0 });
                e.task(Emitter.blocks3D(Emitter.ceilDiv(m, TILE), Emitter.ceilDiv(cols, TILE), splits, MmaInt8Kernels.THREADS),
                        (g, id) -> g.task(id, MmaInt8Kernels::gemm, e.context(), q, w, y, partial, m, cols, kp, splits, sx, zp, offset, sw, sums, bias, hasBias));
                if (splits > 1) {
                    e.task(Emitter.grid1D(m * cols), (g, id) -> g.task(id, MmaInt8Kernels::reduce, e.context(), partial, y, m, cols, splits, sx, zp, offset, sw, sums, bias, hasBias));
                }
            } else {
                e.task(Emitter.grid2D(cols, m, 16), (g, id) -> g.task(id, Int8Kernels::gemm, e.context(), q, w, y, m, cols, k, sx, zp, offset, sw, sums, bias, hasBias));
            }
        }

        @Override
        public void reference(Node n) {
            int m = n.iattr[0], cols = n.iattr[1], k = n.iattr[2], offset = n.iattr[3];
            int lda = padded(k) / 4;
            IntArray q = (IntArray) n.in(0).storage().host();
            float sx = ((FloatArray) n.in(1).storage().host()).get(0);
            int z = (int) ((FloatArray) n.in(2).storage().host()).get(0) - offset;
            Weights w = new Weights(n.in(3), n.in(4), n.in(5), cols, k);
            IntArray packed = (IntArray) n.in(3).storage().host();
            FloatArray sw = (FloatArray) n.in(4).storage().host();
            FloatArray bias = n.iattr[4] != 0 ? (FloatArray) n.in(6).storage().host() : null;
            FloatArray y = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < cols; j++) {
                    long acc = 0;
                    for (int l = 0; l < k; l++) {
                        acc += (long) (unpack(q, i * lda, l) - z) * w.get(packed, j, l);
                    }
                    y.set(i * cols + j, (float) ((double) sx * sw.get(j) * acc + (bias == null ? 0 : bias.get(j))));
                }
            }
        }
    }
}
