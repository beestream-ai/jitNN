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
package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.SparseMatrix;
import ai.beestream.jitnn.Tensor;

/**
 * Graph convolution (Kipf and Welling, 2017), PyTorch Geometric's GCNConv: Â (x W^T) + b, with Â
 * the normalised adjacency {@link SparseMatrix#gcn} of the graph, given with each call since one
 * layer serves many graphs. x [nodes, in] gives [nodes, out].
 */
public final class GraphConv extends Module {

    private final Linear lin;
    private final Tensor bias;
    private SparseMatrix.Valued graph;

    /**
     * Graph convolution from {@code in} to {@code out} features: a linear map without bias initialised from
     * {@code seed}, and a bias of zeros.
     */
    public GraphConv(int in, int out, long seed) {
        this.lin = child(new Linear(in, out, false, seed));
        this.bias = parameter(Tensor.zeros(out));
    }

    /** Sets the graph the next forward calls use. */
    public GraphConv graph(SparseMatrix.Valued adjacency) {
        this.graph = adjacency;
        return this;
    }

    /**
     * Applies the layer to node features x [nodes, in] over the graph's normalised adjacency (see
     * {@link SparseMatrix#gcn}).
     */
    public Tensor forward(Tensor x, SparseMatrix.Valued adjacency) {
        return adjacency.mm(lin.forward(x)).add(bias);
    }

    @Override
    public Tensor forward(Tensor x) {
        if (graph == null) {
            throw new IllegalStateException("GraphConv: set the graph with graph(...) or call forward(x, adjacency)");
        }
        return forward(x, graph);
    }

    /** The weight [out, in]. */
    public Tensor weight() {
        return lin.weight();
    }
}
