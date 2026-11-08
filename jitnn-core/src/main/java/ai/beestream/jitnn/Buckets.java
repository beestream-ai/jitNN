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
package ai.beestream.jitnn;

import java.util.Arrays;

/**
 * A fixed set of sizes that a changing dimension (batch, sequence length) is rounded up to, so
 * requests of many shapes run a few compiled plans instead of compiling one per shape, as
 * TensorRT's optimization profiles or vLLM's CUDA-graph batch sizes do.
 *
 * <p>Padding and cutting happen on the host, before and after the model runs: a pad or a slice
 * on the device would put the real size into that operator's plan key and compile again.
 * {@code ai.beestream.jitnn.nn.Bucketing} builds on this for whole models (masks for padded
 * sequence positions, losses over the real rows only).
 */
public final class Buckets {

    private final int[] sizes;

    private Buckets(int[] sizes) {
        if (sizes.length == 0) {
            throw new IllegalArgumentException("no bucket sizes");
        }
        int[] s = sizes.clone();
        Arrays.sort(s);
        if (s[0] < 1) {
            throw new IllegalArgumentException("bucket sizes must be positive: " + Arrays.toString(sizes));
        }
        this.sizes = s;
    }

    /** These sizes. */
    public static Buckets of(int... sizes) {
        return new Buckets(sizes);
    }

    /** The powers of two from {@code min} up to and including {@code max} (both rounded up to a power of two). */
    public static Buckets powersOfTwo(int min, int max) {
        int v = 1;
        while (v < min) {
            v *= 2;
        }
        int[] s = { v };
        while (v < max) {
            v *= 2;
            s = Arrays.copyOf(s, s.length + 1);
            s[s.length - 1] = v;
        }
        return new Buckets(s);
    }

    /** The multiples of {@code step} up to and including {@code max} (rounded up to a multiple). */
    public static Buckets multiplesOf(int step, int max) {
        int[] s = new int[(max + step - 1) / step];
        for (int i = 0; i < s.length; i++) {
            s[i] = (i + 1) * step;
        }
        return new Buckets(s);
    }

    /** The sizes, ascending. */
    public int[] sizes() {
        return sizes.clone();
    }

    /** The largest size. */
    public int max() {
        return sizes[sizes.length - 1];
    }

    /** The smallest size that holds {@code n}; throws if {@code n} is larger than every size. */
    public int size(int n) {
        for (int s : sizes) {
            if (s >= n) {
                return s;
            }
        }
        throw new IllegalArgumentException(n + " is larger than the largest bucket " + max());
    }

    /**
     * {@code t} with dimension {@code axis} grown to {@code size}, the new elements set to
     * {@code value}; built on the host (a device result is read back first). Returns {@code t}
     * itself when the dimension is already {@code size}. INT32 tensors stay INT32.
     */
    public static Tensor pad(Tensor t, int axis, int size, float value) {
        int[] dims = t.shape().dims();
        int n = dims[axis];
        if (n == size) {
            return t;
        }
        if (n > size) {
            throw new IllegalArgumentException("dimension " + axis + " is " + n + ", larger than " + size);
        }
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) {
            outer *= dims[i];
        }
        for (int i = axis + 1; i < dims.length; i++) {
            inner *= dims[i];
        }
        int[] out = dims.clone();
        out[axis] = size;
        int from = n * inner, to = size * inner;
        if (t.dtype() == DType.I32) {
            int[] src = t.toIntArray();
            int[] dst = new int[outer * to];
            Arrays.fill(dst, (int) value);
            for (int o = 0; o < outer; o++) {
                System.arraycopy(src, o * from, dst, o * to, from);
            }
            return Tensor.of(dst, out);
        }
        float[] src = t.toFloatArray();
        float[] dst = new float[outer * to];
        Arrays.fill(dst, value);
        for (int o = 0; o < outer; o++) {
            System.arraycopy(src, o * from, dst, o * to, from);
        }
        return Tensor.of(dst, out);
    }

    /**
     * The first {@code length} entries of dimension {@code axis} of {@code t}, copied on the host
     * (a device result is read back first). Returns {@code t} itself when nothing is cut.
     */
    public static Tensor cut(Tensor t, int axis, int length) {
        int[] dims = t.shape().dims();
        int n = dims[axis];
        if (n == length) {
            return t;
        }
        if (length > n) {
            throw new IllegalArgumentException("dimension " + axis + " is " + n + ", shorter than " + length);
        }
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) {
            outer *= dims[i];
        }
        for (int i = axis + 1; i < dims.length; i++) {
            inner *= dims[i];
        }
        int[] out = dims.clone();
        out[axis] = length;
        int from = n * inner, to = length * inner;
        if (t.dtype() == DType.I32) {
            int[] src = t.toIntArray();
            int[] dst = new int[outer * to];
            for (int o = 0; o < outer; o++) {
                System.arraycopy(src, o * from, dst, o * to, to);
            }
            return Tensor.of(dst, out);
        }
        float[] src = t.toFloatArray();
        float[] dst = new float[outer * to];
        for (int o = 0; o < outer; o++) {
            System.arraycopy(src, o * from, dst, o * to, to);
        }
        return Tensor.of(dst, out);
    }

    @Override
    public String toString() {
        return "Buckets" + Arrays.toString(sizes);
    }
}
