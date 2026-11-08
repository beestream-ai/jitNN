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
import ai.beestream.jitnn.kernels.GgufKernels;
import ai.beestream.jitnn.kernels.jitllm.TransformerComputeKernelsQ4_0;
import ai.beestream.jitnn.kernels.jitllm.TransformerComputeKernelsQ4_1;
import ai.beestream.jitnn.kernels.jitllm.TransformerComputeKernelsQ4_K;
import ai.beestream.jitnn.kernels.jitllm.TransformerComputeKernelsQ5_K;
import ai.beestream.jitnn.kernels.jitllm.TransformerComputeKernelsQ6_K;

import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Linear layers and dequantisation for weights in GGUF's Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K and Q3_K formats
 * (Q8_0 is {@link Quant}). The products use jitllm's kernels, read straight from the blocks:
 * <ul>
 * <li>one input row (decoding): jitllm's matrix-vector kernel, one work-group per output row, in
 * its warp-shuffle form on CUDA and Metal;
 * <li>up to {@link #TILED_ROWS} rows: jitllm's row-tiled kernel, which decodes each weight once for
 * up to 8 rows (Q4_0, Q4_1, Q5_K);
 * <li>more rows (prefill, batches): the weight dequantised to FP32 and jitNN's GEMM.
 * </ul>
 * The gradient reaches the input (through the dequantised weight), not the quantised weight.
 */
public final class Gguf {

    static final Dequantize DEQUANTIZE = new Dequantize();
    static final MatVec MAT_VEC = new MatVec();

    /** Up to this many input rows use the quantised kernels; beyond, dequantise and GEMM. */
    public static final int TILED_ROWS = Integer.getInteger("jitnn.gguf.tiledRows", 32);
    static final int LOCAL = 128;

    private Gguf() {
    }

    /** Formats with jitllm's matrix-vector kernels; the others dequantise and use the GEMM. */
    static boolean matrixVector(DType t) {
        return t == DType.Q4_0 || t == DType.Q4_1 || t == DType.Q4_K || t == DType.Q5_K || t == DType.Q6_K;
    }

    static boolean tiled(DType t) {
        return t == DType.Q4_0 || t == DType.Q4_1 || t == DType.Q5_K;
    }

    public static Tensor dequantize(Tensor w) {
        GgufKernels.code(w.dtype());
        Tensor y = Tensor.result(w.shape(), DType.F32);
        Tensor.record(DEQUANTIZE, new Tensor[] { w }, new Tensor[] { y }, new int[] { GgufKernels.code(w.dtype()) }, null, true);
        return y;
    }

    /** y = x W^T + b, W [N, K] in a GGUF format, x FP32 [..., K]; b may be null. */
    public static Tensor linear(Tensor x, Tensor w, Tensor bias) {
        Checks.f32(x);
        if (w.dtype() == DType.Q8_0) {
            return Quant.linear(x, w, bias);
        }
        if (w.rank() != 2 || x.dim(-1) != w.dim(1)) {
            throw new IllegalArgumentException(w.dtype() + " linear: input " + x.shape() + " and weight " + w.shape());
        }
        int k = w.dim(1);
        int n = w.dim(0);
        int rows = x.numel() / k;
        int[] dims = x.shape().dims();
        dims[dims.length - 1] = n;
        Tensor y;
        if ((rows == 1 && matrixVector(w.dtype())) || (rows <= TILED_ROWS && tiled(w.dtype()))) {
            y = Tensor.result(Shape.of(rows, n), DType.F32);
            Tensor.record(MAT_VEC, new Tensor[] { x, w }, new Tensor[] { y }, new int[] { rows, n, k, GgufKernels.code(w.dtype()) }, null, true);
            y = y.reshape(dims);
        } else {
            y = x.matmul(dequantize(w), false, true);
        }
        return bias == null ? y : y.add(bias);
    }

    /** inputs [w]; iattr = [format code]. */
    static final class Dequantize implements Op {

        @Override
        public String name() {
            return "gguf_dequantize";
        }

        @Override
        public void emit(Emitter e, Node node) {
            ByteArray w = e.blocks(node.in(0));
            FloatArray y = e.f32(node.out(0));
            int count = node.out(0).numel();
            switch (node.iattr[0]) {
                case GgufKernels.Q4_0 -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ4_0, e.context(), w, y, count));
                case GgufKernels.Q4_1 -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ4_1, e.context(), w, y, count));
                case GgufKernels.Q4_K -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ4_K, e.context(), w, y, count));
                case GgufKernels.Q5_K -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ5_K, e.context(), w, y, count));
                case GgufKernels.Q6_K -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ6_K, e.context(), w, y, count));
                case GgufKernels.Q5_0 -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ5_0, e.context(), w, y, count));
                case GgufKernels.Q5_1 -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ5_1, e.context(), w, y, count));
                case GgufKernels.Q2_K -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ2_K, e.context(), w, y, count));
                default -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, GgufKernels::dequantizeQ3_K, e.context(), w, y, count));
            }
        }

        @Override
        public void reference(Node node) {
            ByteArray w = (ByteArray) node.in(0).storage().host();
            FloatArray y = (FloatArray) node.out(0).storage().host();
            for (int i = 0; i < node.out(0).numel(); i++) {
                y.set(i, GgufKernels.value(w, node.iattr[0], i));
            }
        }
    }

    /** inputs [x, w]; outputs [y [rows, n]]; iattr = [rows, n, k, format code]. */
    static final class MatVec implements Op {

        @Override
        public String name() {
            return "gguf_matvec";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            ByteArray w = e.blocks(node.in(1));
            FloatArray y = e.f32(node.out(0));
            int rows = node.iattr[0], n = node.iattr[1], k = node.iattr[2], code = node.iattr[3];
            boolean warp = e.isCuda() || e.isMetal();
            if (rows == 1) {
                if (warp && code != GgufKernels.Q6_K) {
                    switch (code) {
                        case GgufKernels.Q4_0 -> e.task(Emitter.blocks1D(n, 32), (g, id) -> g.task(id, TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0Simd32, e.context(), x, y, w, k, n));
                        case GgufKernels.Q4_1 -> e.task(Emitter.blocks1D(n, 32), (g, id) -> g.task(id, TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1Simd32, e.context(), x, y, w, k, n));
                        case GgufKernels.Q4_K -> e.task(Emitter.blocks1D(n, 32), (g, id) -> g.task(id, TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_KSimd32, e.context(), x, y, w, k, n));
                        default -> e.task(Emitter.blocks1D(n, 32), (g, id) -> g.task(id, TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_KSimd32, e.context(), x, y, w, k, n));
                    }
                    return;
                }
                switch (code) {
                    case GgufKernels.Q4_0 -> e.task(Emitter.blocks1D(n, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0, e.context(), x, y, w, k, n, LOCAL));
                    case GgufKernels.Q4_1 -> e.task(Emitter.blocks1D(n, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1, e.context(), x, y, w, k, n, LOCAL));
                    case GgufKernels.Q4_K -> e.task(Emitter.blocks1D(n, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K, e.context(), x, y, w, k, n, LOCAL));
                    case GgufKernels.Q5_K -> e.task(Emitter.blocks1D(n, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K, e.context(), x, y, w, k, n, LOCAL));
                    default -> e.task(Emitter.blocks1D(n, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K, e.context(), x, y, w, k, n, LOCAL));
                }
                return;
            }
            switch (code) {
                case GgufKernels.Q4_0 -> {
                    int groups = Emitter.ceilDiv(rows, TransformerComputeKernelsQ4_0.ROW_TILE) * Emitter.ceilDiv(n, TransformerComputeKernelsQ4_0.COL_TILE);
                    e.task(Emitter.blocks1D(groups, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ4_0::matrixVectorTiledBatchQ4_0, e.context(), x, y, w, k, n, rows, LOCAL));
                }
                case GgufKernels.Q4_1 -> {
                    int groups = Emitter.ceilDiv(rows, TransformerComputeKernelsQ4_1.ROW_TILE) * Emitter.ceilDiv(n, TransformerComputeKernelsQ4_1.COL_TILE);
                    e.task(Emitter.blocks1D(groups, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ4_1::matrixVectorTiledBatchQ4_1, e.context(), x, y, w, k, n, rows, LOCAL));
                }
                default -> {
                    int groups = Emitter.ceilDiv(rows, TransformerComputeKernelsQ5_K.ROW_TILE) * Emitter.ceilDiv(n, TransformerComputeKernelsQ5_K.COL_TILE);
                    e.task(Emitter.blocks1D(groups, LOCAL), (g, id) -> g.task(id, TransformerComputeKernelsQ5_K::matrixVectorTiledBatchQ5_K, e.context(), x, y, w, k, n, rows, LOCAL));
                }
            }
        }

        @Override
        public void reference(Node node) {
            int rows = node.iattr[0], n = node.iattr[1], k = node.iattr[2], code = node.iattr[3];
            FloatArray x = (FloatArray) node.in(0).storage().host();
            ByteArray w = (ByteArray) node.in(1).storage().host();
            FloatArray y = (FloatArray) node.out(0).storage().host();
            double[] wr = new double[k];
            for (int c = 0; c < n; c++) {
                for (int j = 0; j < k; j++) {
                    wr[j] = GgufKernels.value(w, code, c * k + j);
                }
                for (int r = 0; r < rows; r++) {
                    double acc = 0;
                    for (int j = 0; j < k; j++) {
                        acc += wr[j] * x.get(r * k + j);
                    }
                    y.set(r * n + c, (float) acc);
                }
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor g) {
            Tensor x = node.in(0);
            Tensor gx = g.matmul(dequantize(node.in(1))).reshape(x.shape().dims());
            return new Tensor[] { gx, null };
        }
    }
}
