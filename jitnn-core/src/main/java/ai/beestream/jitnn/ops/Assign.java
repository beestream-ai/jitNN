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

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Copies an FP32 tensor into an existing persistent tensor of the same size, in place: a value kept
 * on the device between steps (a gradient buffer the next step's optimiser reads, say) without a
 * trip through the host. No gradient.
 */
public final class Assign implements Op {

    static final Assign OP = new Assign();

    private Assign() {
    }

    /** Records dst := src (dst persistent FP32, same number of elements). */
    public static void into(Tensor src, Tensor dst) {
        if (src.dtype() != DType.F32 || dst.dtype() != DType.F32 || src.numel() != dst.numel() || !dst.storage().isPersistent()) {
            throw new IllegalArgumentException("assign: needs FP32 into a persistent FP32 tensor of the same size, got " + src + " into " + dst);
        }
        Tensor.record(OP, new Tensor[] { src }, new Tensor[] { dst }, new int[0], null, true);
    }

    @Override
    public String name() {
        return "assign";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray src = e.f32(n.in(0));
        FloatArray dst = e.f32(n.out(0));
        int count = n.out(0).numel();
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ai.beestream.jitnn.kernels.ElementwiseKernels::copy, e.context(), src, dst, count));
    }

    @Override
    public void reference(Node n) {
        FloatArray src = (FloatArray) n.in(0).storage().host();
        FloatArray dst = (FloatArray) n.out(0).storage().host();
        for (int i = 0; i < n.out(0).numel(); i++) {
            dst.set(i, src.get(i));
        }
    }
}
