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
import ai.beestream.jitnn.kernels.SpatialKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * N-dimensional spatial operators over [N, C, ...spatial] with 1 to 3 spatial axes (see
 * {@link SpatialKernels}): window unfolding (im2col), adaptive pooling and padding.
 */
public final class Spatial {

    static final Unfold UNFOLD = new Unfold();
    static final Fold FOLD = new Fold();
    static final AdaptivePool ADAPTIVE = new AdaptivePool();
    static final AdaptivePoolBackward ADAPTIVE_BACKWARD = new AdaptivePoolBackward();
    static final Pad PAD = new Pad();
    static final PadBackward PAD_BACKWARD = new PadBackward();

    private Spatial() {
    }

    /** The spatial sizes of x [N, C, ...] padded to three axes (leading 1s). */
    static int[] spatial3(Tensor x) {
        int r = x.rank() - 2;
        if (r < 1 || r > 3) {
            throw new IllegalArgumentException("expected [N, C] and 1 to 3 spatial axes, got " + x.shape());
        }
        int[] s = { 1, 1, 1 };
        for (int k = 0; k < r; k++) {
            s[3 - r + k] = x.dim(2 + k);
        }
        return s;
    }

    /** Per-axis values for r spatial axes, padded to three with {@code fill}. */
    static int[] axes3(int[] v, int fill) {
        int[] s = { fill, fill, fill };
        System.arraycopy(v, 0, s, 3 - v.length, v.length);
        return s;
    }

    /**
     * Unfolds the sliding windows of x [N, C, ...] (kernel, stride, padding and dilation per spatial
     * axis) into columns [N, C * prod(kernel), prod(out)], as PyTorch's unfold for any of 1 to 3 axes.
     */
    public static Tensor unfold(Tensor x, int[] kernel, int[] stride, int[] pad, int[] dil) {
        Checks.f32(x);
        int r = x.rank() - 2;
        if (kernel.length != r || stride.length != r || pad.length != r || dil.length != r) {
            throw new IllegalArgumentException("unfold: one kernel/stride/padding/dilation per spatial axis of " + x.shape());
        }
        int[] in = spatial3(x);
        int[] k = axes3(kernel, 1);
        int[] s = axes3(stride, 1);
        int[] p = axes3(pad, 0);
        int[] d = axes3(dil, 1);
        int[] o = new int[3];
        for (int a = 0; a < 3; a++) {
            o[a] = (in[a] + 2 * p[a] - d[a] * (k[a] - 1) - 1) / s[a] + 1;
            if (o[a] < 1) {
                throw new IllegalArgumentException("unfold: empty output along a spatial axis of " + x.shape());
            }
        }
        int c = x.dim(1);
        int[] meta = { c, in[0], in[1], in[2], k[0], k[1], k[2], s[0], s[1], s[2], p[0], p[1], p[2], d[0], d[1], d[2], o[0], o[1], o[2] };
        Tensor cols = Tensor.result(Shape.of(x.dim(0), c * k[0] * k[1] * k[2], o[0] * o[1] * o[2]), DType.F32);
        Tensor.record(UNFOLD, new Tensor[] { x }, new Tensor[] { cols }, meta, null, true);
        return cols;
    }

    /**
     * Folds columns [N, C * prod(kernel), prod(out)] back into x [N, C, ...outputSize], summing
     * overlapping windows: PyTorch's fold, ONNX's Col2Im, the inverse of {@link #unfold} with the
     * same kernel, stride, padding and dilation.
     */
    public static Tensor fold(Tensor cols, int[] outputSize, int[] kernel, int[] stride, int[] pad, int[] dil) {
        Checks.f32(cols);
        Checks.rank(cols, 3);
        int r = outputSize.length;
        if (r < 1 || r > 3 || kernel.length != r || stride.length != r || pad.length != r || dil.length != r) {
            throw new IllegalArgumentException("fold: one output size/kernel/stride/padding/dilation per spatial axis (1 to 3)");
        }
        int[] in = axes3(outputSize, 1);
        int[] k = axes3(kernel, 1);
        int[] s = axes3(stride, 1);
        int[] p = axes3(pad, 0);
        int[] d = axes3(dil, 1);
        int[] o = new int[3];
        for (int a = 0; a < 3; a++) {
            o[a] = (in[a] + 2 * p[a] - d[a] * (k[a] - 1) - 1) / s[a] + 1;
        }
        int window = k[0] * k[1] * k[2];
        if (cols.dim(1) % window != 0 || cols.dim(2) != o[0] * o[1] * o[2]) {
            throw new IllegalArgumentException("fold: columns " + cols.shape() + " do not match the kernel and output size");
        }
        int c = cols.dim(1) / window;
        int[] meta = { c, in[0], in[1], in[2], k[0], k[1], k[2], s[0], s[1], s[2], p[0], p[1], p[2], d[0], d[1], d[2], o[0], o[1], o[2] };
        int[] dims = new int[2 + r];
        dims[0] = cols.dim(0);
        dims[1] = c;
        System.arraycopy(outputSize, 0, dims, 2, r);
        Tensor x = Tensor.result(Shape.of(dims), DType.F32);
        Tensor.record(FOLD, new Tensor[] { cols }, new Tensor[] { x }, meta, null, true);
        return x;
    }

    static final class Unfold implements Op {

        @Override
        public String name() {
            return "unfold";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            FloatArray cols = e.f32(n.out(0));
            IntArray m = e.constant(n.iattr);
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, SpatialKernels::unfold, e.context(), x, cols, m, total));
        }

        @Override
        public void reference(Node n) {
            int[] m = n.iattr;
            FloatArray x = (FloatArray) n.in(0).storage().host();
            FloatArray cols = (FloatArray) n.out(0).storage().host();
            int c = m[0];
            int l = m[16] * m[17] * m[18];
            int rows = c * m[4] * m[5] * m[6];
            int batch = n.in(0).dim(0);
            for (int b = 0; b < batch; b++) {
                for (int row = 0; row < rows; row++) {
                    int jx = row % m[6], jy = (row / m[6]) % m[5], jz = (row / (m[6] * m[5])) % m[4], ch = row / (m[6] * m[5] * m[4]);
                    for (int pos = 0; pos < l; pos++) {
                        int ox = pos % m[18], oy = (pos / m[18]) % m[17], oz = pos / (m[18] * m[17]);
                        int iz = oz * m[7] - m[10] + jz * m[13];
                        int iy = oy * m[8] - m[11] + jy * m[14];
                        int ix = ox * m[9] - m[12] + jx * m[15];
                        boolean inside = iz >= 0 && iz < m[1] && iy >= 0 && iy < m[2] && ix >= 0 && ix < m[3];
                        cols.set((b * rows + row) * l + pos, inside ? x.get((((b * c + ch) * m[1] + iz) * m[2] + iy) * m[3] + ix) : 0);
                    }
                }
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor gx = Tensor.result(n.in(0).shape(), DType.F32);
            Tensor.record(FOLD, new Tensor[] { gy }, new Tensor[] { gx }, n.iattr.clone(), null, true);
            return new Tensor[] { gx };
        }
    }

    /** Fold (col2im), also the gradient of unfold: inputs [cols], output x; iattr as {@link Unfold}. */
    static final class Fold implements Op {

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor gc = Tensor.result(n.in(0).shape(), DType.F32);
            Tensor.record(UNFOLD, new Tensor[] { gy }, new Tensor[] { gc }, n.iattr.clone(), null, true);
            return new Tensor[] { gc };
        }

        @Override
        public String name() {
            return "fold";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray gc = e.f32(n.in(0));
            FloatArray gx = e.f32(n.out(0));
            IntArray m = e.constant(n.iattr);
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, SpatialKernels::fold, e.context(), gc, gx, m, total));
        }

        @Override
        public void reference(Node n) {
            int[] m = n.iattr;
            FloatArray gc = (FloatArray) n.in(0).storage().host();
            FloatArray gx = (FloatArray) n.out(0).storage().host();
            double[] acc = new double[n.out(0).numel()];
            int c = m[0];
            int l = m[16] * m[17] * m[18];
            int rows = c * m[4] * m[5] * m[6];
            int batch = n.out(0).dim(0);
            for (int b = 0; b < batch; b++) {
                for (int row = 0; row < rows; row++) {
                    int jx = row % m[6], jy = (row / m[6]) % m[5], jz = (row / (m[6] * m[5])) % m[4], ch = row / (m[6] * m[5] * m[4]);
                    for (int pos = 0; pos < l; pos++) {
                        int ox = pos % m[18], oy = (pos / m[18]) % m[17], oz = pos / (m[18] * m[17]);
                        int iz = oz * m[7] - m[10] + jz * m[13];
                        int iy = oy * m[8] - m[11] + jy * m[14];
                        int ix = ox * m[9] - m[12] + jx * m[15];
                        if (iz >= 0 && iz < m[1] && iy >= 0 && iy < m[2] && ix >= 0 && ix < m[3]) {
                            acc[(((b * c + ch) * m[1] + iz) * m[2] + iy) * m[3] + ix] += gc.get((b * rows + row) * l + pos);
                        }
                    }
                }
            }
            for (int i = 0; i < acc.length; i++) {
                gx.set(i, (float) acc[i]);
            }
        }
    }

    /** Adaptive average or max pooling of x [N, C, ...] to {@code out} (one size per spatial axis). */
    public static Tensor adaptivePool(Tensor x, int[] out, boolean max) {
        Checks.f32(x);
        int r = x.rank() - 2;
        if (out.length != r) {
            throw new IllegalArgumentException("adaptive pooling: one output size per spatial axis of " + x.shape());
        }
        int[] in = spatial3(x);
        int[] o = axes3(out, 1);
        int[] dims = x.shape().dims();
        System.arraycopy(out, 0, dims, 2, r);
        Tensor y = Tensor.result(Shape.of(dims), DType.F32);
        int[] meta = { in[0], in[1], in[2], o[0], o[1], o[2], max ? 1 : 0 };
        if (max) {
            Tensor arg = Tensor.result(y.shape(), DType.I32);
            Tensor.record(ADAPTIVE, new Tensor[] { x }, new Tensor[] { y, arg }, meta, null, true);
        } else {
            Tensor.record(ADAPTIVE, new Tensor[] { x }, new Tensor[] { y }, meta, null, true);
        }
        return y;
    }

    /** iattr = [D, H, W, OD, OH, OW, max]; outputs [y] or [y, argmax]. */
    static final class AdaptivePool implements Op {

        @Override
        public String name() {
            return "adaptive_pool";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            FloatArray y = e.f32(n.out(0));
            int mode = n.iattr[6];
            IntArray arg = mode != 0 ? e.i32(n.out(1)) : e.constant(0);
            IntArray m = e.constant(java.util.Arrays.copyOf(n.iattr, 6));
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, SpatialKernels::adaptivePool, e.context(), x, y, arg, m, mode, total));
        }

        @Override
        public void reference(Node n) {
            int[] m = n.iattr;
            FloatArray x = (FloatArray) n.in(0).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            IntArray arg = m[6] != 0 ? (IntArray) n.out(1).storage().host() : null;
            for (int i = 0; i < n.out(0).numel(); i++) {
                int ox = i % m[5], oy = (i / m[5]) % m[4], oz = (i / (m[5] * m[4])) % m[3], p = i / (m[5] * m[4] * m[3]);
                double acc = 0;
                double best = Double.NEGATIVE_INFINITY;
                int at = 0;
                int count = 0;
                for (int z = SpatialKernels.binStart(oz, m[0], m[3]); z < SpatialKernels.binEnd(oz, m[0], m[3]); z++) {
                    for (int yy = SpatialKernels.binStart(oy, m[1], m[4]); yy < SpatialKernels.binEnd(oy, m[1], m[4]); yy++) {
                        for (int xx = SpatialKernels.binStart(ox, m[2], m[5]); xx < SpatialKernels.binEnd(ox, m[2], m[5]); xx++) {
                            int idx = ((p * m[0] + z) * m[1] + yy) * m[2] + xx;
                            double v = x.get(idx);
                            acc += v;
                            count++;
                            if (v > best) {
                                best = v;
                                at = idx;
                            }
                        }
                    }
                }
                if (arg == null) {
                    y.set(i, (float) (acc / count));
                } else {
                    y.set(i, (float) best);
                    arg.set(i, at);
                }
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor gx = Tensor.result(n.in(0).shape(), DType.F32);
            Tensor[] in = n.iattr[6] != 0 ? new Tensor[] { gy, n.out(1) } : new Tensor[] { gy };
            Tensor.record(ADAPTIVE_BACKWARD, in, new Tensor[] { gx }, n.iattr.clone(), null, true);
            return new Tensor[] { gx };
        }
    }

    static final class AdaptivePoolBackward implements Op {

        @Override
        public String name() {
            return "adaptive_pool_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray gy = e.f32(n.in(0));
            int mode = n.iattr[6];
            IntArray arg = mode != 0 ? e.i32(n.in(1)) : e.constant(0);
            FloatArray gx = e.f32(n.out(0));
            IntArray m = e.constant(java.util.Arrays.copyOf(n.iattr, 6));
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, SpatialKernels::adaptivePoolBackward, e.context(), gy, arg, gx, m, mode, total));
        }

        @Override
        public void reference(Node n) {
            int[] m = n.iattr;
            FloatArray gy = (FloatArray) n.in(0).storage().host();
            IntArray arg = m[6] != 0 ? (IntArray) n.in(1).storage().host() : null;
            FloatArray gx = (FloatArray) n.out(0).storage().host();
            double[] acc = new double[n.out(0).numel()];
            for (int o = 0; o < n.in(0).numel(); o++) {
                int ox = o % m[5], oy = (o / m[5]) % m[4], oz = (o / (m[5] * m[4])) % m[3], p = o / (m[5] * m[4] * m[3]);
                if (arg != null) {
                    acc[arg.get(o)] += gy.get(o);
                    continue;
                }
                int z0 = SpatialKernels.binStart(oz, m[0], m[3]), z1 = SpatialKernels.binEnd(oz, m[0], m[3]);
                int y0 = SpatialKernels.binStart(oy, m[1], m[4]), y1 = SpatialKernels.binEnd(oy, m[1], m[4]);
                int x0 = SpatialKernels.binStart(ox, m[2], m[5]), x1 = SpatialKernels.binEnd(ox, m[2], m[5]);
                double share = gy.get(o) / (double) ((z1 - z0) * (y1 - y0) * (x1 - x0));
                for (int z = z0; z < z1; z++) {
                    for (int yy = y0; yy < y1; yy++) {
                        for (int xx = x0; xx < x1; xx++) {
                            acc[((p * m[0] + z) * m[1] + yy) * m[2] + xx] += share;
                        }
                    }
                }
            }
            for (int i = 0; i < acc.length; i++) {
                gx.set(i, (float) acc[i]);
            }
        }
    }

    /**
     * Pads the spatial axes of x [N, C, ...]: {@code before[k]} and {@code after[k]} for spatial axis
     * k; mode {@link SpatialKernels#CONSTANT} (with {@code value}), REFLECT or REPLICATE.
     */
    public static Tensor pad(Tensor x, int[] before, int[] after, int mode, float value) {
        Checks.f32(x);
        int r = x.rank() - 2;
        if (before.length != r || after.length != r) {
            throw new IllegalArgumentException("pad: one before/after pair per spatial axis of " + x.shape());
        }
        int[] in = spatial3(x);
        int[] b = axes3(before, 0);
        int[] a = axes3(after, 0);
        int[] dims = x.shape().dims();
        for (int k = 0; k < r; k++) {
            int size = x.dim(2 + k);
            if (mode == SpatialKernels.REFLECT && (before[k] >= size || after[k] >= size)) {
                throw new IllegalArgumentException("reflect padding must be smaller than the axis it pads: " + x.shape());
            }
            dims[2 + k] = size + before[k] + after[k];
        }
        int[] o = { in[0] + b[0] + a[0], in[1] + b[1] + a[1], in[2] + b[2] + a[2] };
        Tensor y = Tensor.result(Shape.of(dims), DType.F32);
        int[] meta = { in[0], in[1], in[2], o[0], o[1], o[2], b[0], b[1], b[2], mode };
        Tensor.record(PAD, new Tensor[] { x }, new Tensor[] { y }, meta, new float[] { value }, true);
        return y;
    }

    static int source(int o, int before, int in, int mode) {
        return SpatialKernels.padSource(o, before, in, mode);
    }

    /** iattr = [D, H, W, OD, OH, OW, before D, before H, before W, mode]; fattr = [value]. */
    static final class Pad implements Op {

        @Override
        public String name() {
            return "pad";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            FloatArray y = e.f32(n.out(0));
            IntArray m = e.constant(java.util.Arrays.copyOf(n.iattr, 9));
            int mode = n.iattr[9];
            float value = n.fattr[0];
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, SpatialKernels::pad, e.context(), x, y, m, mode, value, total));
        }

        @Override
        public void reference(Node n) {
            int[] m = n.iattr;
            FloatArray x = (FloatArray) n.in(0).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                int ox = i % m[5], oy = (i / m[5]) % m[4], oz = (i / (m[5] * m[4])) % m[3], p = i / (m[5] * m[4] * m[3]);
                int sz = source(oz, m[6], m[0], m[9]), sy = source(oy, m[7], m[1], m[9]), sx = source(ox, m[8], m[2], m[9]);
                y.set(i, sz >= 0 && sy >= 0 && sx >= 0 ? x.get(((p * m[0] + sz) * m[1] + sy) * m[2] + sx) : n.fattr[0]);
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor gx = Tensor.result(n.in(0).shape(), DType.F32);
            Tensor.record(PAD_BACKWARD, new Tensor[] { gy }, new Tensor[] { gx }, n.iattr.clone(), null, true);
            return new Tensor[] { gx };
        }
    }

    static final class PadBackward implements Op {

        @Override
        public String name() {
            return "pad_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray gy = e.f32(n.in(0));
            FloatArray gx = e.f32(n.out(0));
            IntArray m = e.constant(java.util.Arrays.copyOf(n.iattr, 9));
            int mode = n.iattr[9];
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, SpatialKernels::padBackward, e.context(), gy, gx, m, mode, total));
        }

        @Override
        public void reference(Node n) {
            int[] m = n.iattr;
            FloatArray gy = (FloatArray) n.in(0).storage().host();
            FloatArray gx = (FloatArray) n.out(0).storage().host();
            double[] acc = new double[n.out(0).numel()];
            for (int o = 0; o < n.in(0).numel(); o++) {
                int ox = o % m[5], oy = (o / m[5]) % m[4], oz = (o / (m[5] * m[4])) % m[3], p = o / (m[5] * m[4] * m[3]);
                int sz = source(oz, m[6], m[0], m[9]), sy = source(oy, m[7], m[1], m[9]), sx = source(ox, m[8], m[2], m[9]);
                if (sz >= 0 && sy >= 0 && sx >= 0) {
                    acc[((p * m[0] + sz) * m[1] + sy) * m[2] + sx] += gy.get(o);
                }
            }
            for (int i = 0; i < acc.length; i++) {
                gx.set(i, (float) acc[i]);
            }
        }
    }
}
