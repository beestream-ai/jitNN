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
import ai.beestream.jitnn.kernels.Q8Kernels;
import ai.beestream.jitnn.kernels.jitllm.NativeProjectionKernels;

import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Q8_0 quantisation, dequantisation and linear layers on Q8_0 weights (see {@link Q8Kernels}). */
public final class Quant {

    static final Quantize QUANTIZE = new Quantize();
    static final Dequantize DEQUANTIZE = new Dequantize();
    static final LinearQ8 LINEAR = new LinearQ8();

    private Quant() {
    }

    /** x (FP32, last axis a multiple of 32) to Q8_0, block by block along the last axis. */
    public static Tensor quantizeQ8(Tensor x) {
        Checks.f32(x);
        if (x.dim(-1) % Q8Kernels.BLOCK != 0) {
            throw new IllegalArgumentException("Q8_0 needs a last axis that is a multiple of 32, got " + x.shape());
        }
        Tensor q = Tensor.result(x.shape(), DType.Q8_0);
        Tensor.record(QUANTIZE, new Tensor[] { x }, new Tensor[] { q }, null, null, true);
        return q;
    }

    /** Q8_0 to FP32. */
    public static Tensor dequantize(Tensor q) {
        if (q.dtype() != DType.Q8_0) {
            throw new IllegalArgumentException("dequantize needs a Q8_0 tensor, got " + q.dtype());
        }
        Tensor x = Tensor.result(q.shape(), DType.F32);
        Tensor.record(DEQUANTIZE, new Tensor[] { q }, new Tensor[] { x }, null, null, true);
        return x;
    }

    /**
     * y = x W^T + b with W Q8_0 [N, K] (PyTorch layout) and x FP32 [..., K]; b may be null. The
     * gradient reaches x (through the dequantised weight), not the quantised weight.
     */
    public static Tensor linear(Tensor x, Tensor w, Tensor bias) {
        Checks.f32(x);
        if (w.dtype() != DType.Q8_0 || w.rank() != 2 || x.dim(-1) != w.dim(1)) {
            throw new IllegalArgumentException("Q8_0 linear: input " + x.shape() + " and weight " + w.shape() + " " + w.dtype());
        }
        int k = w.dim(1);
        int n = w.dim(0);
        int m = x.numel() / k;
        int[] dims = x.shape().dims();
        dims[dims.length - 1] = n;
        Tensor y = Tensor.result(Shape.of(dims), DType.F32);
        Tensor[] in = bias == null ? new Tensor[] { x, w } : new Tensor[] { x, w, bias };
        Tensor.record(LINEAR, in, new Tensor[] { y }, new int[] { m, n, k, bias == null ? 0 : 1 }, null, true);
        return y;
    }

    static float half(float v) {
        return new HalfFloat(v).getFloat32();
    }

    static final class Quantize implements Op {

        @Override
        public String name() {
            return "quantize_q8_0";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            ByteArray q = e.q8(n.out(0));
            int blocks = n.in(0).numel() / Q8Kernels.BLOCK;
            e.task(Emitter.blocks1D(blocks, Q8Kernels.BLOCK), (g, id) -> g.task(id, Q8Kernels::quantize, e.context(), x, q));
        }

        @Override
        public void reference(Node n) {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            ByteArray q = (ByteArray) n.out(0).storage().host();
            for (int b = 0; b < n.in(0).numel() / Q8Kernels.BLOCK; b++) {
                double amax = 0;
                for (int l = 0; l < Q8Kernels.BLOCK; l++) {
                    amax = Math.max(amax, Math.abs(x.get(b * Q8Kernels.BLOCK + l)));
                }
                float d = (float) (amax / 127.0);
                q.setHalfFloat(b * Q8Kernels.BLOCK_BYTES, new HalfFloat(d));
                for (int l = 0; l < Q8Kernels.BLOCK; l++) {
                    double s = d > 0 ? x.get(b * Q8Kernels.BLOCK + l) / (double) d : 0;
                    long v = Math.max(-127, Math.min(127, s >= 0 ? (long) (s + 0.5) : -(long) (-s + 0.5)));
                    q.set(b * Q8Kernels.BLOCK_BYTES + 2 + l, (byte) v);
                }
            }
        }
    }

    static final class Dequantize implements Op {

        @Override
        public String name() {
            return "dequantize_q8_0";
        }

        @Override
        public void emit(Emitter e, Node n) {
            ByteArray q = e.q8(n.in(0));
            FloatArray x = e.f32(n.out(0));
            int count = n.out(0).numel();
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, NativeProjectionKernels::dequantizeQ8_0ToFP32, e.context(), q, x, count));
        }

        @Override
        public void reference(Node n) {
            ByteArray q = (ByteArray) n.in(0).storage().host();
            FloatArray x = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                int block = i / Q8Kernels.BLOCK * Q8Kernels.BLOCK_BYTES;
                x.set(i, q.getHalfFloat(block).getFloat32() * q.get(block + 2 + i % Q8Kernels.BLOCK));
            }
        }
    }

    /** iattr = [M, N, K, hasBias]. */
    static final class LinearQ8 implements Op {

        @Override
        public String name() {
            return "linear_q8_0";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int m = n.iattr[0], rows = n.iattr[1], k = n.iattr[2], hasBias = n.iattr[3];
            FloatArray x = e.f32(n.in(0));
            ByteArray w = e.q8(n.in(1));
            FloatArray bias = hasBias != 0 ? e.f32(n.in(2)) : e.dummyF32();
            FloatArray y = e.f32(n.out(0));
            if (e.isCuda() || e.isMetal()) {
                e.task(Emitter.blocks2D(Emitter.ceilDiv(rows, Q8Kernels.WARP_ROWS), m, Q8Kernels.WARP_ROWS * 32, 1),
                        (g, id) -> g.task(id, Q8Kernels::linearWarp, e.context(), x, w, bias, y, k, rows, hasBias));
            } else {
                e.task(Emitter.blocks2D(rows, m, Q8Kernels.GROUP, 1), (g, id) -> g.task(id, Q8Kernels::linear, e.context(), x, w, bias, y, k, rows, hasBias));
            }
        }

        @Override
        public void reference(Node n) {
            int m = n.iattr[0], rows = n.iattr[1], k = n.iattr[2];
            FloatArray x = (FloatArray) n.in(0).storage().host();
            ByteArray w = (ByteArray) n.in(1).storage().host();
            FloatArray b = n.iattr[3] != 0 ? (FloatArray) n.in(2).storage().host() : null;
            FloatArray y = (FloatArray) n.out(0).storage().host();
            int blocks = k / Q8Kernels.BLOCK;
            for (int i = 0; i < m; i++) {
                for (int r = 0; r < rows; r++) {
                    double acc = b == null ? 0 : b.get(r);
                    for (int j = 0; j < k; j++) {
                        int off = (r * blocks + j / Q8Kernels.BLOCK) * Q8Kernels.BLOCK_BYTES;
                        acc += (double) w.getHalfFloat(off).getFloat32() * w.get(off + 2 + j % Q8Kernels.BLOCK) * x.get(i * k + j);
                    }
                    y.set(i * rows + r, (float) acc);
                }
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor x = n.in(0);
            int k = n.iattr[2];
            Tensor gx = x.requiresGrad() ? gy.reshape(-1, n.iattr[1]).matmul(dequantize(n.in(1))).reshape(x.shape().dims()) : null;
            Tensor gb = n.iattr[3] != 0 && n.in(2).requiresGrad() ? gy.reshape(-1, n.iattr[1]).sum(0, false) : null;
            return n.iattr[3] != 0 ? new Tensor[] { gx, null, gb } : new Tensor[] { gx, null };
        }
    }
}
