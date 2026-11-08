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
import ai.beestream.jitnn.kernels.ElementwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Slicing and concatenation along one axis. A tensor is seen as [outer, axis, inner]; every op
 * here moves contiguous runs of {@code inner} elements with {@link ElementwiseKernels#copyBlock}.
 */
public final class Slice {

    static final Narrow NARROW = new Narrow();
    static final NarrowGrad NARROW_GRAD = new NarrowGrad();
    static final Concat CONCAT = new Concat();

    private Slice() {
    }

    /** Elements {@code start .. start + length - 1} of {@code axis}. */
    public static Tensor narrow(Tensor x, int axis, int start, int length) {
        Checks.f32(x);
        int ax = x.shape().axis(axis);
        int size = x.dim(ax);
        if (start < 0 || length < 0 || start + length > size) {
            throw new IllegalArgumentException("narrow(" + axis + ", " + start + ", " + length + ") out of range for " + x.shape());
        }
        if (start == 0 && length == size) {
            return x;
        }
        int[] dims = x.shape().dims();
        dims[ax] = length;
        Tensor out = Tensor.result(Shape.of(dims), DType.F32);
        int[] oi = outerInner(x.shape(), ax);
        Tensor.record(NARROW, new Tensor[] { x }, new Tensor[] { out }, new int[] { oi[0], size, length, oi[1], start, ax }, null, true);
        return out;
    }

    /** Joins {@code parts} along {@code axis}; all other dimensions must agree. */
    public static Tensor concat(int axis, Tensor... parts) {
        if (parts.length == 0) {
            throw new IllegalArgumentException("concat needs at least one tensor");
        }
        Checks.f32(parts);
        if (parts.length == 1) {
            return parts[0];
        }
        Shape first = parts[0].shape();
        int ax = first.axis(axis);
        int total = 0;
        int[] iattr = new int[4 + parts.length];
        for (int p = 0; p < parts.length; p++) {
            Shape s = parts[p].shape();
            if (s.rank() != first.rank()) {
                throw new IllegalArgumentException("concat of " + first + " and " + s);
            }
            for (int d = 0; d < s.rank(); d++) {
                if (d != ax && s.dim(d) != first.dim(d)) {
                    throw new IllegalArgumentException("concat along " + axis + " of " + first + " and " + s);
                }
            }
            iattr[4 + p] = s.dim(ax);
            total += s.dim(ax);
        }
        int[] dims = first.dims();
        dims[ax] = total;
        Shape outShape = Shape.of(dims);
        int[] oi = outerInner(outShape, ax);
        iattr[0] = oi[0];
        iattr[1] = total;
        iattr[2] = oi[1];
        iattr[3] = ax;
        Tensor out = Tensor.result(outShape, DType.F32);
        Tensor.record(CONCAT, parts.clone(), new Tensor[] { out }, iattr, null, true);
        return out;
    }

    /** [product of dims before axis, product of dims after axis]. */
    static int[] outerInner(Shape s, int ax) {
        int outer = 1;
        for (int d = 0; d < ax; d++) {
            outer *= s.dim(d);
        }
        int inner = 1;
        for (int d = ax + 1; d < s.rank(); d++) {
            inner *= s.dim(d);
        }
        return new int[] { outer, inner };
    }

    static void copyBlockHost(FloatArray x, FloatArray out, int outer, int inLen, int outLen, int len, int inner, int inOff, int outOff) {
        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < len; j++) {
                for (int t = 0; t < inner; t++) {
                    out.set((o * outLen + outOff + j) * inner + t, x.get((o * inLen + inOff + j) * inner + t));
                }
            }
        }
    }

    static void emitCopy(Emitter e, FloatArray x, FloatArray out, int outer, int inLen, int outLen, int len, int inner, int inOff, int outOff) {
        int count = outer * len * inner;
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ElementwiseKernels::copyBlock, e.context(), x, out, len, inner, inLen, outLen, inOff, outOff, count));
    }

    /** iattr = [outer, inLen, length, inner, start, axis]. */
    static final class Narrow implements Op {

        @Override
        public String name() {
            return "narrow";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            emitCopy(e, e.f32(n.in(0)), e.f32(n.out(0)), a[0], a[1], a[2], a[2], a[3], a[4], 0);
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            copyBlockHost((FloatArray) n.in(0).storage().host(), (FloatArray) n.out(0).storage().host(), a[0], a[1], a[2], a[2], a[3], a[4], 0);
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor dx = Tensor.result(n.in(0).shape(), DType.F32);
            Tensor.record(NARROW_GRAD, new Tensor[] { gy }, new Tensor[] { dx }, n.iattr.clone(), null, true);
            return new Tensor[] { dx };
        }
    }

    /** Scatters dy back into a zero tensor of the input's shape. iattr as {@link Narrow}. */
    static final class NarrowGrad implements Op {

        @Override
        public String name() {
            return "narrow_grad";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            FloatArray dy = e.f32(n.in(0));
            FloatArray dx = e.f32(n.out(0));
            int total = n.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ElementwiseKernels::fill, e.context(), dx, 0.0f, total));
            emitCopy(e, dy, dx, a[0], a[2], a[1], a[2], a[3], 0, a[4]);
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            FloatArray dx = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                dx.set(i, 0.0f);
            }
            copyBlockHost((FloatArray) n.in(0).storage().host(), dx, a[0], a[2], a[1], a[2], a[3], 0, a[4]);
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            return new Tensor[] { narrow(gy, n.iattr[5], n.iattr[4], n.iattr[2]) };
        }
    }

    /** iattr = [outer, total, inner, axis, len0, len1, ...]. */
    static final class Concat implements Op {

        @Override
        public String name() {
            return "concat";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            FloatArray out = e.f32(n.out(0));
            int off = 0;
            for (int p = 0; p < n.inputs.length; p++) {
                int len = a[4 + p];
                emitCopy(e, e.f32(n.in(p)), out, a[0], len, a[1], len, a[2], 0, off);
                off += len;
            }
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            FloatArray out = (FloatArray) n.out(0).storage().host();
            int off = 0;
            for (int p = 0; p < n.inputs.length; p++) {
                int len = a[4 + p];
                copyBlockHost((FloatArray) n.in(p).storage().host(), out, a[0], len, a[1], len, a[2], 0, off);
                off += len;
            }
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            int[] a = n.iattr;
            int ax = a[3];
            Tensor[] gx = new Tensor[n.inputs.length];
            int off = 0;
            for (int p = 0; p < gx.length; p++) {
                int len = a[4 + p];
                gx[p] = n.in(p).requiresGrad() ? narrow(gy, ax, off, len) : null;
                off += len;
            }
            return gx;
        }
    }
}
