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
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.ScanKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Cumulative sums and products, and stable sorting, along one axis (see {@link ScanKernels}). */
public final class Ordered {

    static final Scan SCAN = new Scan();
    static final CumprodBackward CUMPROD_BACKWARD = new CumprodBackward();
    static final Sort SORT = new Sort();
    static final SortBackward SORT_BACKWARD = new SortBackward();

    private Ordered() {
    }

    /** [outer, n, inner] for {@code axis} of x. */
    static int[] lines(Tensor x, int axis) {
        int ax = x.shape().axis(axis);
        int outer = 1;
        int inner = 1;
        for (int d = 0; d < ax; d++) {
            outer *= x.dim(d);
        }
        for (int d = ax + 1; d < x.rank(); d++) {
            inner *= x.dim(d);
        }
        return new int[] { outer, x.dim(ax), inner };
    }

    /** Inclusive cumulative sum (mode SUM) or product (PROD) along {@code axis}. */
    public static Tensor scan(Tensor x, int axis, int mode) {
        return scan(x, axis, mode, false);
    }

    static Tensor scan(Tensor x, int axis, int mode, boolean reverse) {
        Checks.f32(x);
        int[] l = lines(x, axis);
        Tensor y = Tensor.result(x.shape(), DType.F32);
        Tensor.record(SCAN, new Tensor[] { x }, new Tensor[] { y }, new int[] { l[0], l[1], l[2], mode, reverse ? 1 : 0, x.shape().axis(axis) }, null, true);
        return y;
    }

    /** Sorted values, original INT32 indices and each element's sorted position (rank), along {@code axis}. */
    static Tensor[] sortWithRanks(Tensor x, int axis, boolean descending) {
        Checks.f32(x);
        int[] l = lines(x, axis);
        Tensor values = Tensor.result(x.shape(), DType.F32);
        Tensor indices = Tensor.result(x.shape(), DType.I32);
        Tensor ranks = Tensor.result(x.shape(), DType.I32);
        Tensor.record(SORT, new Tensor[] { x }, new Tensor[] { values, indices, ranks }, new int[] { l[0], l[1], l[2], descending ? 1 : 0 }, null, true);
        return new Tensor[] { values, indices.detach(), ranks.detach() };
    }

    /** Sorted values and their original INT32 indices along {@code axis}, stable. */
    public static Tensor[] sort(Tensor x, int axis, boolean descending) {
        Checks.f32(x);
        int[] l = lines(x, axis);
        Tensor values = Tensor.result(x.shape(), DType.F32);
        Tensor indices = Tensor.result(x.shape(), DType.I32);
        Tensor ranks = Tensor.result(x.shape(), DType.I32);
        Tensor.record(SORT, new Tensor[] { x }, new Tensor[] { values, indices, ranks }, new int[] { l[0], l[1], l[2], descending ? 1 : 0 }, null, true);
        return new Tensor[] { values, indices.detach() };
    }

    /** iattr = [outer, n, inner, mode, reverse, axis]. */
    static final class Scan implements Op {

        @Override
        public String name() {
            return "scan";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            FloatArray y = e.f32(node.out(0));
            int n = node.iattr[1], inner = node.iattr[2], mode = node.iattr[3], reverse = node.iattr[4];
            int lines = node.iattr[0] * inner;
            e.task(Emitter.grid1D(lines), (g, id) -> g.task(id, ScanKernels::scan, e.context(), x, y, n, inner, lines, mode, reverse));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            FloatArray y = (FloatArray) node.out(0).storage().host();
            int outer = node.iattr[0], n = node.iattr[1], inner = node.iattr[2];
            boolean sum = node.iattr[3] == ScanKernels.SUM;
            for (int o = 0; o < outer; o++) {
                for (int q = 0; q < inner; q++) {
                    double acc = sum ? 0 : 1;
                    for (int t = 0; t < n; t++) {
                        int k = node.iattr[4] != 0 ? n - 1 - t : t;
                        int at = (o * n + k) * inner + q;
                        acc = sum ? acc + x.get(at) : acc * x.get(at);
                        y.set(at, (float) acc);
                    }
                }
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor gy) {
            int axis = node.iattr[5];
            if (node.iattr[3] == ScanKernels.SUM) {
                return new Tensor[] { scan(gy, axis, ScanKernels.SUM, node.iattr[4] == 0) };
            }
            if (node.iattr[4] != 0) {
                throw new UnsupportedOperationException("no gradient for a reversed cumulative product");
            }
            Tensor dx = Tensor.result(node.in(0).shape(), DType.F32);
            Tensor.record(CUMPROD_BACKWARD, new Tensor[] { node.in(0), gy }, new Tensor[] { dx }, node.iattr.clone(), null, true);
            return new Tensor[] { dx };
        }
    }

    static final class CumprodBackward implements Op {

        @Override
        public String name() {
            return "cumprod_bwd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            FloatArray dy = e.f32(node.in(1));
            FloatArray dx = e.f32(node.out(0));
            int n = node.iattr[1], inner = node.iattr[2];
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ScanKernels::cumprodBackward, e.context(), x, dy, dx, n, inner, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            FloatArray dy = (FloatArray) node.in(1).storage().host();
            FloatArray dx = (FloatArray) node.out(0).storage().host();
            int outer = node.iattr[0], n = node.iattr[1], inner = node.iattr[2];
            for (int o = 0; o < outer; o++) {
                for (int q = 0; q < inner; q++) {
                    for (int i = 0; i < n; i++) {
                        double acc = 0;
                        for (int j = i; j < n; j++) {
                            double prod = 1;
                            for (int k = 0; k <= j; k++) {
                                if (k != i) {
                                    prod *= x.get((o * n + k) * inner + q);
                                }
                            }
                            acc += dy.get((o * n + j) * inner + q) * prod;
                        }
                        dx.set((o * n + i) * inner + q, (float) acc);
                    }
                }
            }
        }
    }

    /** iattr = [outer, n, inner, descending]; outputs [values, indices, ranks]. */
    static final class Sort implements Op {

        @Override
        public String name() {
            return "sort";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray x = e.f32(node.in(0));
            FloatArray values = e.f32(node.out(0));
            IntArray indices = e.i32(node.out(1));
            IntArray ranks = e.i32(node.out(2));
            int n = node.iattr[1], inner = node.iattr[2], descending = node.iattr[3];
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ScanKernels::sortByRank, e.context(), x, values, indices, ranks, n, inner, descending, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray x = (FloatArray) node.in(0).storage().host();
            FloatArray values = (FloatArray) node.out(0).storage().host();
            IntArray indices = (IntArray) node.out(1).storage().host();
            IntArray ranks = (IntArray) node.out(2).storage().host();
            int outer = node.iattr[0], n = node.iattr[1], inner = node.iattr[2];
            boolean descending = node.iattr[3] != 0;
            for (int o = 0; o < outer; o++) {
                for (int q = 0; q < inner; q++) {
                    final int oo = o, qq = q;
                    Integer[] order = new Integer[n];
                    for (int k = 0; k < n; k++) {
                        order[k] = k;
                    }
                    java.util.Arrays.sort(order, (a, b) -> {
                        int c = Float.compare(x.get((oo * n + a) * inner + qq), x.get((oo * n + b) * inner + qq));
                        return descending ? -c : c;
                    });
                    for (int r = 0; r < n; r++) {
                        int k = order[r];
                        values.set((o * n + r) * inner + q, x.get((o * n + k) * inner + q));
                        indices.set((o * n + r) * inner + q, k);
                        ranks.set((o * n + k) * inner + q, r);
                    }
                }
            }
        }

        @Override
        public Tensor[] backward(Node node, Tensor gy) {
            Tensor dx = Tensor.result(node.in(0).shape(), DType.F32);
            Tensor.record(SORT_BACKWARD, new Tensor[] { gy, node.out(2) }, new Tensor[] { dx }, node.iattr.clone(), null, true);
            return new Tensor[] { dx };
        }
    }

    static final class SortBackward implements Op {

        @Override
        public String name() {
            return "sort_bwd";
        }

        @Override
        public void emit(Emitter e, Node node) {
            FloatArray dy = e.f32(node.in(0));
            IntArray ranks = e.i32(node.in(1));
            FloatArray dx = e.f32(node.out(0));
            int n = node.iattr[1], inner = node.iattr[2];
            int total = node.out(0).numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ScanKernels::sortBackward, e.context(), dy, ranks, dx, n, inner, total));
        }

        @Override
        public void reference(Node node) {
            FloatArray dy = (FloatArray) node.in(0).storage().host();
            IntArray ranks = (IntArray) node.in(1).storage().host();
            FloatArray dx = (FloatArray) node.out(0).storage().host();
            int n = node.iattr[1], inner = node.iattr[2];
            for (int i = 0; i < node.out(0).numel(); i++) {
                int q = i % inner;
                int o = i / (inner * n);
                dx.set(i, dy.get(o * n * inner + ranks.get(i) * inner + q));
            }
        }
    }
}
