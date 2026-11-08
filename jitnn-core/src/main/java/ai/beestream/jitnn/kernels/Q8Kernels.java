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
 *
 * Parts of this file are taken from jitllm and remain under the MIT License:
 *
 * MIT License
 *
 * Copyright (c) 2025 Beehive lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Q8_0 (GGUF's 8-bit format: per 32 consecutive values an FP16 scale d and 32 signed bytes q, value
 * d q, 34 bytes) quantisation and products, adapted from jitllm's kernels: the quantiser from
 * TransformerComputeKernelsQ4_0.quantizeActivationQ8Blocks (writing GGUF blocks instead of packed
 * ints), the portable product from TransformerComputeKernelsLayered.matrixVectorGenericQ8Byte and
 * the warp product from Gemma4Kernels.matrixVectorQ8_0Warp (both with input rows and a bias added).
 * Dequantisation is jitllm's NativeProjectionKernels.dequantizeQ8_0ToFP32 as is.
 */
public final class Q8Kernels {

    public static final int BLOCK = 32;
    public static final int BLOCK_BYTES = 34;
    /** Threads of a portable product's work-group (one output per group). */
    public static final int GROUP = 128;
    /** Output rows per warp-product work-group (one warp each). */
    public static final int WARP_ROWS = 8;

    private Q8Kernels() {
    }

    /**
     * One work-group of 32 per block: d = max |x| / 127, q = round(x / d) (halves away from zero, as
     * ggml), a block of zeros gets d = 0 and q = 0.
     */
    public static void quantize(KernelContext kc, FloatArray x, ByteArray q) {
        int block = kc.groupIdx;
        int lane = kc.localIdx;
        float[] shared = kc.allocateFloatLocalArray(BLOCK);
        float value = x.get(block * BLOCK + lane);
        shared[lane] = TornadoMath.abs(value);
        kc.localBarrier();
        for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                shared[lane] = TornadoMath.max(shared[lane], shared[lane + stride]);
            }
            kc.localBarrier();
        }
        float maxAbs = shared[0];
        float d = maxAbs / 127.0f;
        float inverse = maxAbs > 0.0f ? 1.0f / d : 0.0f;
        float scaled = value * inverse;
        int v = (int) (scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        v = TornadoMath.min(127, TornadoMath.max(-127, v));
        int off = block * BLOCK_BYTES;
        if (lane == 0) {
            q.setHalfFloat(off, new HalfFloat(d));
        }
        q.set(off + 2 + lane, (byte) v);
    }

    /**
     * y[m, row] = x[m, :] . dequant(w[row, :]) (+ bias[row]) for x [M, K], w Q8_0 [N, K]: one
     * work-group of {@link #GROUP} threads per (row, m), partial sums in a shared-memory tree.
     */
    public static void linear(KernelContext kc, FloatArray x, ByteArray w, FloatArray bias, FloatArray y, int k, int rows, int hasBias) {
        int row = kc.groupIdx;
        int m = kc.groupIdy;
        int lid = kc.localIdx;
        float[] sums = kc.allocateFloatLocalArray(GROUP);
        int blocks = k / BLOCK;
        int rowBase = row * blocks * BLOCK_BYTES;
        int xBase = m * k;
        float acc = 0.0f;
        for (int j = lid; j < k; j += GROUP) {
            int off = rowBase + (j / BLOCK) * BLOCK_BYTES;
            acc += w.getHalfFloat(off).getFloat32() * w.get(off + 2 + j % BLOCK) * x.get(xBase + j);
        }
        sums[lid] = acc;
        kc.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                sums[lid] = sums[lid] + sums[lid + stride];
            }
            kc.localBarrier();
        }
        if (lid == 0) {
            y.set(m * rows + row, sums[0] + (hasBias != 0 ? bias.get(row) : 0.0f));
        }
    }

    /**
     * As {@link #linear} with one warp per output row ({@link #WARP_ROWS} per group), lane l taking
     * element l of every block, summed by shuffles (CUDA and Metal).
     */
    public static void linearWarp(KernelContext kc, FloatArray x, ByteArray w, FloatArray bias, FloatArray y, int k, int rows, int hasBias) {
        int warp = kc.localIdx >> 5;
        int lane = kc.localIdx & 31;
        int row = kc.groupIdx * WARP_ROWS + warp;
        int m = kc.groupIdy;
        if (row < rows) {
            int blocks = k / BLOCK;
            int rowBase = row * blocks * BLOCK_BYTES;
            int xBase = m * k;
            float sum = 0.0f;
            for (int b = 0; b < blocks; b++) {
                int off = rowBase + b * BLOCK_BYTES;
                sum += w.getHalfFloat(off).getFloat32() * w.get(off + 2 + lane) * x.get(xBase + b * BLOCK + lane);
            }
            sum += kc.simdShuffleDown(sum, 16);
            sum += kc.simdShuffleDown(sum, 8);
            sum += kc.simdShuffleDown(sum, 4);
            sum += kc.simdShuffleDown(sum, 2);
            sum += kc.simdShuffleDown(sum, 1);
            if (lane == 0) {
                y.set(m * rows + row, sum + (hasBias != 0 ? bias.get(row) : 0.0f));
            }
        }
    }
}
