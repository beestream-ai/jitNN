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
package ai.beestream.jitnn.natives;

import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Native;
import ai.beestream.jitnn.engine.Native.Lib;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * MLX calls as TornadoVM library tasks on Metal (MLX's own Metal kernels, launched on TornadoVM's
 * buffers). MLX checks that each array holds exactly the elements its shape arguments describe,
 * so these calls are only made for arrays of that exact size (see {@link Libs}).
 */
final class Mlx {

    private Mlx() {
    }

    /** C [m, n] = A [m, k] B [k, n]. */
    static void matmul(Emitter e, FloatArray a, FloatArray b, FloatArray c, int m, int n, int k) {
        e.library(Lib.MLX, "gemm", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::matmul, a, b, c, m, k, n));
    }

    /** C [m, n] = A [m, k] W^T for W [n, k] (a linear layer's weight). */
    static void matmulTransposed(Emitter e, FloatArray a, FloatArray w, FloatArray c, int m, int n, int k) {
        e.library(Lib.MLX, "gemm", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::matmulTransposed, a, w, c, m, k, n));
    }

    /** FP16 C [m, n] = A [m, k] B [k, n] for FP16 A and B. */
    static void matmulHalf(Emitter e, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int m, int n, int k) {
        e.library(Lib.MLX, "gemmHalf", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::matmul, a, b, c, m, k, n));
    }

    /** FP16 C [m, n] = A [m, k] W^T for FP16 A and W [n, k]. */
    static void matmulTransposedHalf(Emitter e, HalfFloatArray a, HalfFloatArray w, HalfFloatArray c, int m, int n, int k) {
        e.library(Lib.MLX, "gemmHalf", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::matmulTransposed, a, w, c, m, k, n));
    }

    /** C[z] [m, n] = A[z] [m, k] B[z] [k, n] for z < batch, all contiguous. */
    static void matmulBatched(Emitter e, FloatArray a, FloatArray b, FloatArray c, int batch, int m, int n, int k) {
        e.library(Lib.MLX, "gemm", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::einsumBatchedMatmul, a, b, c, batch, m, k, n));
    }

    /** Convolution, channels-last: x [n, h, w, c], weight [o, k, k, c], out [n, oh, ow, o]; one group, no dilation. */
    static void conv2d(Emitter e, FloatArray x, FloatArray w, FloatArray out, int n, int h, int wd, int c, int o, int k, int stride, int pad) {
        e.library(Lib.MLX, "conv", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::conv2d, x, w, out, n, h, wd, c, o, k, k, stride, pad, 1, 1));
    }

    /** {@link #conv2d} on FP16 arrays. */
    static void conv2dHalf(Emitter e, HalfFloatArray x, HalfFloatArray w, HalfFloatArray out, int n, int h, int wd, int c, int o, int k, int stride, int pad) {
        e.library(Lib.MLX, "convHalf", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::conv2d, x, w, out, n, h, wd, c, o, k, k, stride, pad, 1, 1));
    }

    /** Row softmax of x [rows, cols]. */
    static void softmax(Emitter e, FloatArray x, FloatArray y, int rows, int cols) {
        e.library(Lib.MLX, "softmax", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::softmaxRows, x, y, rows, cols));
    }

    /** y = LayerNorm(x [rows, cols]) * weight + bias. */
    static void layerNorm(Emitter e, FloatArray x, FloatArray w, FloatArray b, FloatArray y, int rows, int cols, float eps) {
        e.library(Lib.MLX, "norm", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::layerNorm, x, w, b, y, rows, cols, eps));
    }

    /** y = RMSNorm(x [rows, cols]) * weight. */
    static void rmsNorm(Emitter e, FloatArray x, FloatArray w, FloatArray y, int rows, int cols, float eps) {
        e.library(Lib.MLX, "norm", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::rmsNorm, x, w, y, rows, cols, eps));
    }

    /** o = softmax(scale q k^T) v for q [batch, heads, t, d], k and v [batch, heads, s, d]. */
    static void attention(Emitter e, FloatArray q, FloatArray k, FloatArray v, FloatArray o, int batch, int heads, int t, int s, int d, float scale, boolean causal) {
        e.library(Lib.MLX, "sdpa", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::scaledDotProductAttention, q, k, v, o, batch, heads, heads, t, s, d, scale, causal));
    }

    /** {@link #attention} on FP16 arrays. */
    static void attentionHalf(Emitter e, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, HalfFloatArray o, int batch, int heads, int t, int s, int d, float scale,
            boolean causal) {
        e.library(Lib.MLX, "sdpaHalf", (g, id) -> g.libraryTask(id, uk.ac.manchester.tornado.mlx.Mlx::scaledDotProductAttention, q, k, v, o, batch, heads, heads, t, s, d, scale, causal));
    }

    /** Runs a small product on the current device: null if MLX works there, otherwise why not. */
    static String probe() {
        // without MLX's metallib (brew install mlx, or -Dtornado.mlx.metallib) the probe fails to run;
        // an 8 x 8 identity times B gives B (MLX rejects 1 x 1 outputs); MLX needs an M4-class GPU or newer
        int s = 8;
        FloatArray a = new FloatArray(s * s);
        FloatArray b = new FloatArray(s * s);
        for (int i = 0; i < s; i++) {
            a.set(i * s + i, 1);
            for (int j = 0; j < s; j++) {
                b.set(i * s + j, i * s + j);
            }
        }
        FloatArray out = new FloatArray(s * s);
        TaskGraph g = Native.probeGraph("MLX").transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
                .libraryTask("t", uk.ac.manchester.tornado.mlx.Mlx::matmul, a, b, out, s, s, s).transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        Native.runProbe(g);
        for (int i = 0; i < s * s; i++) {
            if (out.get(i) != i) {
                return "probe returned " + out.get(i) + " for " + i;
            }
        }
        return null;
    }
}
