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

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.Fp8Kernels;

import uk.ac.manchester.tornado.api.types.FP8;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * FP8 tensors with a per-tensor scale, as PyTorch's float8 types are used with torch._scaled_mm:
 * quantize divides by scale = amax / (largest finite FP8 value) and rounds, dequantize multiplies
 * back, and the GEMMs take FP8 operands with their scales and accumulate in FP32. Runs on every
 * backend (see {@link Fp8Kernels}); FP8 tensor-core instructions are not used.
 */
public final class Fp8 {

    static final Quantize QUANTIZE = new Quantize();
    static final Dequantize DEQUANTIZE = new Dequantize();
    static final Gemm GEMM = new Gemm();

    private Fp8() {
    }

    static int format(DType t) {
        return switch (t) {
            case FP8_E4M3 -> Fp8Kernels.E4M3;
            case FP8_E5M2 -> Fp8Kernels.E5M2;
            default -> throw new IllegalArgumentException("not an FP8 type: " + t);
        };
    }

    /** [NaN, +infinity] for the decoders (see {@link Fp8Kernels}). */
    static Tensor special() {
        return Tensor.of(new float[] { Float.NaN, Float.POSITIVE_INFINITY }, 2);
    }

    static float max(DType t) {
        return t == DType.FP8_E4M3 ? FP8.E4M3_MAX : FP8.E5M2_MAX;
    }

    /** The scale [1] that maps x's largest magnitude to the largest finite value of {@code type}. */
    public static Tensor scaleFor(Tensor x, DType type) {
        return x.abs().reshape(x.numel()).max(0, true).maximum(Tensor.scalar(1e-12f)).div(Tensor.scalar(max(type)));
    }

    /** [x quantized to {@code type}, its scale [1]], with the scale from x's largest magnitude. */
    public static Tensor[] quantize(Tensor x, DType type) {
        Tensor scale = scaleFor(x, type);
        return new Tensor[] { quantize(x, scale, type), scale };
    }

    /** x / scale rounded to {@code type} (round to nearest even, saturating). */
    public static Tensor quantize(Tensor x, Tensor scale, DType type) {
        Checks.f32(x, scale);
        format(type);
        Tensor q = Tensor.result(x.shape(), type);
        Tensor.record(QUANTIZE, new Tensor[] { x, scale }, new Tensor[] { q }, new int[] { format(type) }, null, true);
        return q;
    }

    public static Tensor dequantize(Tensor q, Tensor scale) {
        Tensor y = Tensor.result(q.shape(), DType.F32);
        Tensor.record(DEQUANTIZE, new Tensor[] { q, scale, special() }, new Tensor[] { y }, new int[] { format(q.dtype()) }, null, true);
        return y;
    }

    /**
     * Quantize then dequantize, with the straight-through gradient (identity), for training that
     * simulates FP8: x + (dequantize(quantize(x)) - x) with the second term detached.
     */
    public static Tensor fakeQuantize(Tensor x, DType type) {
        return fakeQuantize(x, scaleFor(x, type).detach(), type);
    }

    /** {@link #fakeQuantize(Tensor, DType)} with a given scale [1]. */
    public static Tensor fakeQuantize(Tensor x, Tensor scale, DType type) {
        Tensor rounded = dequantize(quantize(x.detach(), scale, type), scale);
        return x.add(rounded.sub(x).detach());
    }

    /**
     * y = a W^T (+ bias), with W FP8 [m, k] scaled by sw and a either FP32 [..., k] (sa null) or
     * FP8 scaled by sa (torch._scaled_mm). The gradient reaches an FP32 a and the bias.
     */
    public static Tensor linear(Tensor a, Tensor sa, Tensor w, Tensor sw, Tensor bias) {
        boolean aFp8 = a.dtype() != DType.F32;
        if (w.rank() != 2 || a.dim(-1) != w.dim(1)) {
            throw new IllegalArgumentException("FP8 linear: input " + a.shape() + " and weight " + w.shape());
        }
        int k = w.dim(1);
        int m = w.dim(0);
        int rows = a.numel() / k;
        int[] dims = a.shape().dims();
        dims[dims.length - 1] = m;
        Tensor y = Tensor.result(Shape.of(dims), DType.F32);
        // unused operands are passed existing arrays of the right type
        Tensor x = aFp8 ? sw : a;
        Tensor xq = aFp8 ? a : w;
        Tensor.record(GEMM, new Tensor[] { x, xq, aFp8 ? sa : sw, w, sw, bias != null ? bias : sw, special() }, new Tensor[] { y },
                new int[] { rows, m, k, aFp8 ? format(a.dtype()) : -1, format(w.dtype()), bias != null ? 1 : 0 }, null, true);
        return y;
    }

    static final class Quantize implements Op {

        @Override
        public String name() {
            return "fp8_quantize";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            FloatArray s = e.f32(n.in(1));
            ByteArray q = e.fp8(n.out(0));
            int count = n.out(0).numel(), format = n.iattr[0];
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, Fp8Kernels::quantize, e.context(), x, s, q, format, count));
        }

        @Override
        public void reference(Node n) {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            FloatArray s = (FloatArray) n.in(1).storage().host();
            ByteArray q = (ByteArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                q.set(i, Fp8Kernels.encode(x.get(i) / s.get(0), n.iattr[0]));
            }
        }
    }

    static final class Dequantize implements Op {

        @Override
        public String name() {
            return "fp8_dequantize";
        }

        @Override
        public void emit(Emitter e, Node n) {
            ByteArray q = e.fp8(n.in(0));
            FloatArray s = e.f32(n.in(1));
            FloatArray sp = e.f32(n.in(2));
            FloatArray y = e.f32(n.out(0));
            int count = n.out(0).numel(), format = n.iattr[0];
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, Fp8Kernels::dequantize, e.context(), q, s, sp, y, format, count));
        }

        @Override
        public void reference(Node n) {
            ByteArray q = (ByteArray) n.in(0).storage().host();
            FloatArray s = (FloatArray) n.in(1).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                y.set(i, Fp8Kernels.decode(q.get(i), n.iattr[0], Float.NaN, Float.POSITIVE_INFINITY) * s.get(0));
            }
        }
    }

    /** inputs [x, xq, sa, w, sw, bias, special]; iattr = [rows, m, k, aFormat (-1: FP32), wFormat, hasBias]. */
    static final class Gemm implements Op {

        @Override
        public String name() {
            return "fp8_gemm";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            ByteArray xq = e.fp8(n.in(1));
            FloatArray sa = e.f32(n.in(2));
            ByteArray w = e.fp8(n.in(3));
            FloatArray sw = e.f32(n.in(4));
            FloatArray bias = e.f32(n.in(5));
            FloatArray sp = e.f32(n.in(6));
            FloatArray y = e.f32(n.out(0));
            int[] a = n.iattr;
            int rows = a[0], m = a[1], k = a[2], af = a[3], wf = a[4], hb = a[5];
            e.task(Emitter.grid2D(m, rows, Fp8Kernels.TILE), (g, id) -> g.task(id, Fp8Kernels::gemm, e.context(), x, xq, sa, w, sw, bias, sp, y, rows, m, k, af, wf, hb));
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            int rows = a[0], m = a[1], k = a[2], af = a[3], wf = a[4];
            FloatArray x = (FloatArray) n.in(0).storage().host();
            ByteArray xq = (ByteArray) n.in(1).storage().host();
            float sa = af >= 0 ? ((FloatArray) n.in(2).storage().host()).get(0) : 1.0f;
            ByteArray w = (ByteArray) n.in(3).storage().host();
            float sw = ((FloatArray) n.in(4).storage().host()).get(0);
            FloatArray bias = (FloatArray) n.in(5).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < m; c++) {
                    double acc = 0;
                    for (int i = 0; i < k; i++) {
                        double av = af >= 0 ? Fp8Kernels.decode(xq.get(r * k + i), af, Float.NaN, Float.POSITIVE_INFINITY) : x.get(r * k + i);
                        acc += av * Fp8Kernels.decode(w.get(c * k + i), wf, Float.NaN, Float.POSITIVE_INFINITY);
                    }
                    y.set(r * m + c, (float) (acc * sa * sw + (a[5] == 1 ? bias.get(c) : 0)));
                }
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor g) {
            int[] a = n.iattr;
            int m = a[1], k = a[2];
            Tensor g2 = g.reshape(a[0], m);
            Tensor gx = a[3] < 0 ? g2.matmul(dequantize(n.in(3), n.in(4))).reshape(n.in(0).shape().dims()) : null;
            Tensor gb = a[5] == 1 ? g2.sum(0, false) : null;
            return new Tensor[] { gx, null, null, null, null, gb, null };
        }
    }
}
