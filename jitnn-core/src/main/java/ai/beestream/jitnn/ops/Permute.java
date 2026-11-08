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
import ai.beestream.jitnn.JitnnConfig;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.ElementwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Reorders dimensions into a new contiguous tensor. iattr = the permutation. */
public final class Permute implements Op {

    static final Permute OP = new Permute();

    private Permute() {
    }

    public static Tensor apply(Tensor x, int... axes) {
        Checks.f32(x);
        int[] order = axes.clone();
        int r = x.rank();
        if (order.length != r) {
            throw new IllegalArgumentException("permutation of length " + order.length + " for rank " + r);
        }
        boolean[] seen = new boolean[r];
        int[] dims = new int[r];
        boolean identity = true;
        for (int i = 0; i < r; i++) {
            int a = x.shape().axis(order[i]);
            if (seen[a]) {
                throw new IllegalArgumentException("repeated axis in permutation");
            }
            seen[a] = true;
            dims[i] = x.dim(a);
            identity &= a == i;
            order[i] = a;
        }
        if (identity) {
            return x;
        }
        Tensor out = Tensor.result(Shape.of(dims), DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, order.clone(), null, true);
        return out;
    }

    static int[] meta(Shape in, int[] order) {
        int r = order.length;
        int[] strides = in.strides();
        int[] dims = in.dims();
        int[] meta = new int[1 + 2 * r];
        meta[0] = r;
        for (int d = 0; d < r; d++) {
            meta[1 + d] = dims[order[d]];
            meta[1 + r + d] = strides[order[d]];
        }
        return meta;
    }

    @Override
    public String name() {
        return "permute";
    }

    /** True when the permutation only swaps the last two axes (a batched matrix transpose). */
    static boolean isLastTwoSwap(int[] order) {
        int r = order.length;
        if (r < 2 || order[r - 1] != r - 2 || order[r - 2] != r - 1) {
            return false;
        }
        for (int i = 0; i < r - 2; i++) {
            if (order[i] != i) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray out = e.f32(n.out(0));
        if (isLastTwoSwap(n.iattr)) {
            int rows = n.in(0).dim(-2);
            int cols = n.in(0).dim(-1);
            int batch = n.in(0).numel() / Math.max(rows * cols, 1);
            String variant = JitnnConfig.current().get(JitnnConfig.TRANSPOSE);
            if (variant.equals("32")) {
                int tilesY = Emitter.ceilDiv(rows, 32);
                e.task(Emitter.blocks2D(Emitter.ceilDiv(cols, 32), tilesY * batch, 32, 8),
                        (g, id) -> g.task(id, ElementwiseKernels::transposeTiled32, e.context(), x, out, rows, cols, tilesY));
                return;
            }
            if (variant.equals("16")) {
                int tilesY = Emitter.ceilDiv(rows, ElementwiseKernels.TT);
                int tilesX = Emitter.ceilDiv(cols, ElementwiseKernels.TT);
                e.task(Emitter.blocks2D(tilesX, tilesY * batch, ElementwiseKernels.TT, ElementwiseKernels.TT),
                        (g, id) -> g.task(id, ElementwiseKernels::transposeTiled, e.context(), x, out, rows, cols, tilesY));
                return;
            }
        }
        IntArray meta = e.constant(meta(n.in(0).shape(), n.iattr));
        int count = n.out(0).numel();
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::permute, e.context(), x, out, meta, count));
    }

    @Override
    public void reference(Node n) {
        int[] meta = meta(n.in(0).shape(), n.iattr);
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray o = (FloatArray) n.out(0).storage().host();
        int r = meta[0];
        for (int i = 0; i < n.out(0).numel(); i++) {
            int rem = i;
            int src = 0;
            for (int d = r - 1; d >= 0; d--) {
                int c = rem % meta[1 + d];
                rem /= meta[1 + d];
                src += c * meta[1 + r + d];
            }
            o.set(i, x.get(src));
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        int[] inverse = new int[n.iattr.length];
        for (int i = 0; i < inverse.length; i++) {
            inverse[n.iattr[i]] = i;
        }
        return new Tensor[] { gy.permute(inverse) };
    }
}
