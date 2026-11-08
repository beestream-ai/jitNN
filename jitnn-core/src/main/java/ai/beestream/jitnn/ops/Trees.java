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
import ai.beestream.jitnn.kernels.TreeKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Operators for histogram-based decision trees (see {@link TreeKernels}); none is differentiable. */
public final class Trees {

    static final Bin BIN = new Bin();
    static final Histogram HISTOGRAM = new Histogram();
    static final Route ROUTE = new Route();
    static final Predict PREDICT = new Predict();

    private Trees() {
    }

    /** INT32 bins [n, F] of x [n, F] against ascending edges [F, E]. */
    public static Tensor bin(Tensor x, Tensor edges) {
        Checks.f32(x);
        Checks.rank(x, 2);
        Tensor out = Tensor.result(x.shape(), DType.I32);
        Tensor.record(BIN, new Tensor[] { x, edges }, new Tensor[] { out }, new int[0], null, true);
        return out;
    }

    /** Histograms [nodes, F, binCount, C] of stats [n, C] per node slot (INT32 node [n], -1 inactive). */
    public static Tensor histogram(Tensor bins, Tensor node, Tensor stats, int nodes, int binCount) {
        Checks.f32(stats);
        Tensor out = Tensor.result(Shape.of(nodes, bins.dim(1), binCount, stats.dim(1)), DType.F32);
        Tensor.record(HISTOGRAM, new Tensor[] { bins, node, stats }, new Tensor[] { out }, new int[] { nodes, binCount }, null, true);
        return out;
    }

    /** The next level's node slot of each sample, from per-slot feature (-1: leaf), split bin and left child slot. */
    public static Tensor route(Tensor bins, Tensor node, Tensor feature, Tensor split, Tensor child) {
        Tensor out = Tensor.result(node.shape(), DType.I32);
        Tensor.record(ROUTE, new Tensor[] { bins, node, feature, split, child }, new Tensor[] { out }, new int[0], null, true);
        return out;
    }

    /** Sum over trees of the reached leaves' values [T, M, O]: [n, O]; trees as feature, split, child [T, M]. */
    public static Tensor predict(Tensor bins, Tensor feature, Tensor split, Tensor child, Tensor value) {
        Tensor out = Tensor.result(Shape.of(bins.dim(0), value.dim(2)), DType.F32);
        Tensor.record(PREDICT, new Tensor[] { bins, feature, split, child, value }, new Tensor[] { out }, new int[0], null, true);
        return out;
    }

    private static int[] ints(Tensor t) {
        IntArray a = (IntArray) t.storage().host();
        int[] v = new int[t.shape().numel()];
        for (int i = 0; i < v.length; i++) {
            v[i] = a.get(i);
        }
        return v;
    }

    static final class Bin implements Op {

        @Override
        public String name() {
            return "tree_bin";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray x = e.f32(n.in(0));
            FloatArray edges = e.f32(n.in(1));
            IntArray bins = e.i32(n.out(0));
            int features = n.in(0).dim(1), edgeCount = n.in(1).dim(1), total = n.in(0).shape().numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, TreeKernels::bin, e.context(), x, edges, bins, features, edgeCount, total));
        }

        @Override
        public void reference(Node n) {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            FloatArray edges = (FloatArray) n.in(1).storage().host();
            IntArray bins = (IntArray) n.out(0).storage().host();
            int features = n.in(0).dim(1), edgeCount = n.in(1).dim(1);
            for (int i = 0; i < n.in(0).shape().numel(); i++) {
                int f = i % features;
                int b = 0;
                while (b < edgeCount && edges.get(f * edgeCount + b) < x.get(i)) {
                    b++;
                }
                bins.set(i, b);
            }
        }
    }

    static final class Histogram implements Op {

        @Override
        public String name() {
            return "tree_histogram";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray bins = e.i32(n.in(0));
            IntArray node = e.i32(n.in(1));
            FloatArray stats = e.f32(n.in(2));
            FloatArray hist = e.f32(n.out(0));
            int samples = n.in(0).dim(0), features = n.in(0).dim(1), binCount = n.iattr[1], channels = n.in(2).dim(1), total = n.out(0).shape().numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, TreeKernels::histogram, e.context(), bins, node, stats, hist, samples, features, binCount, channels, total));
        }

        @Override
        public void reference(Node n) {
            int[] bins = ints(n.in(0));
            int[] node = ints(n.in(1));
            FloatArray stats = (FloatArray) n.in(2).storage().host();
            FloatArray hist = (FloatArray) n.out(0).storage().host();
            int samples = n.in(0).dim(0), features = n.in(0).dim(1), binCount = n.iattr[1], channels = n.in(2).dim(1);
            double[] acc = new double[n.out(0).shape().numel()];
            for (int s = 0; s < samples; s++) {
                if (node[s] < 0) {
                    continue;
                }
                for (int f = 0; f < features; f++) {
                    for (int c = 0; c < channels; c++) {
                        acc[((node[s] * features + f) * binCount + bins[s * features + f]) * channels + c] += stats.get(s * channels + c);
                    }
                }
            }
            for (int i = 0; i < acc.length; i++) {
                hist.set(i, (float) acc[i]);
            }
        }
    }

    static final class Route implements Op {

        @Override
        public String name() {
            return "tree_route";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray bins = e.i32(n.in(0));
            IntArray node = e.i32(n.in(1));
            IntArray feature = e.i32(n.in(2));
            IntArray split = e.i32(n.in(3));
            IntArray child = e.i32(n.in(4));
            IntArray out = e.i32(n.out(0));
            int features = n.in(0).dim(1), samples = n.in(0).dim(0);
            e.task(Emitter.grid1D(samples), (g, id) -> g.task(id, TreeKernels::route, e.context(), bins, node, feature, split, child, out, features, samples));
        }

        @Override
        public void reference(Node n) {
            int[] bins = ints(n.in(0));
            int[] node = ints(n.in(1));
            int[] feature = ints(n.in(2));
            int[] split = ints(n.in(3));
            int[] child = ints(n.in(4));
            IntArray out = (IntArray) n.out(0).storage().host();
            int features = n.in(0).dim(1);
            for (int s = 0; s < node.length; s++) {
                int nd = node[s];
                int r = -1;
                if (nd >= 0 && feature[nd] >= 0) {
                    r = child[nd] + (bins[s * features + feature[nd]] > split[nd] ? 1 : 0);
                }
                out.set(s, r);
            }
        }
    }

    static final class Predict implements Op {

        @Override
        public String name() {
            return "tree_predict";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray bins = e.i32(n.in(0));
            IntArray feature = e.i32(n.in(1));
            IntArray split = e.i32(n.in(2));
            IntArray child = e.i32(n.in(3));
            FloatArray value = e.f32(n.in(4));
            FloatArray out = e.f32(n.out(0));
            int features = n.in(0).dim(1), trees = n.in(1).dim(0), maxNodes = n.in(1).dim(1), outputs = n.in(4).dim(2), total = n.out(0).shape().numel();
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, TreeKernels::predict, e.context(), bins, feature, split, child, value, out, features, trees, maxNodes, outputs, total));
        }

        @Override
        public void reference(Node n) {
            int[] bins = ints(n.in(0));
            int[] feature = ints(n.in(1));
            int[] split = ints(n.in(2));
            int[] child = ints(n.in(3));
            FloatArray value = (FloatArray) n.in(4).storage().host();
            FloatArray out = (FloatArray) n.out(0).storage().host();
            int samples = n.in(0).dim(0), features = n.in(0).dim(1), trees = n.in(1).dim(0), maxNodes = n.in(1).dim(1), outputs = n.in(4).dim(2);
            for (int s = 0; s < samples; s++) {
                for (int o = 0; o < outputs; o++) {
                    double acc = 0;
                    for (int t = 0; t < trees; t++) {
                        int base = t * maxNodes;
                        int nd = 0;
                        while (feature[base + nd] >= 0) {
                            nd = child[base + nd] + (bins[s * features + feature[base + nd]] > split[base + nd] ? 1 : 0);
                        }
                        acc += value.get((base + nd) * outputs + o);
                    }
                    out.set(s * outputs + o, (float) acc);
                }
            }
        }
    }
}
