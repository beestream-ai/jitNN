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
import ai.beestream.jitnn.kernels.BitwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Bitwise and, or, xor, shifts, integer remainders (with broadcasting) and not, on INT32 tensors; iattr = [op] (not: none). No gradients. */
public final class Bitwise implements Op {

    public enum Kind {
        AND(BitwiseKernels.AND),
        OR(BitwiseKernels.OR),
        XOR(BitwiseKernels.XOR),
        SHIFT_LEFT(BitwiseKernels.SHIFT_LEFT),
        SHIFT_RIGHT(BitwiseKernels.SHIFT_RIGHT),
        REMAINDER(BitwiseKernels.REMAINDER),
        FMOD(BitwiseKernels.FMOD);

        final int code;

        Kind(int code) {
            this.code = code;
        }
    }

    static final Bitwise OP = new Bitwise();
    static final Not NOT = new Not();

    private Bitwise() {
    }

    private static void int32(Tensor t) {
        if (t.dtype() != DType.I32) {
            throw new IllegalArgumentException("bitwise operations take INT32 tensors, got " + t);
        }
    }

    public static Tensor apply(Kind kind, Tensor a, Tensor b) {
        int32(a);
        int32(b);
        Tensor out = Tensor.result(Shape.broadcast(a.shape(), b.shape()), DType.I32);
        Tensor.record(OP, new Tensor[] { a, b }, new Tensor[] { out }, new int[] { kind.code }, null, true);
        return out;
    }

    public static Tensor not(Tensor x) {
        int32(x);
        Tensor out = Tensor.result(x.shape(), DType.I32);
        Tensor.record(NOT, new Tensor[] { x }, new Tensor[] { out }, new int[0], null, true);
        return out;
    }

    @Override
    public String name() {
        return "bitwise";
    }

    @Override
    public void emit(Emitter e, Node n) {
        IntArray a = e.i32(n.in(0));
        IntArray b = e.i32(n.in(1));
        IntArray out = e.i32(n.out(0));
        IntArray meta = e.constant(Binary.broadcastMeta(n.in(0).shape(), n.in(1).shape(), n.out(0).shape()));
        int op = n.iattr[0];
        int count = n.out(0).numel();
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, BitwiseKernels::binary, e.context(), a, b, out, meta, op, count));
    }

    @Override
    public void reference(Node n) {
        int[] meta = Binary.broadcastMeta(n.in(0).shape(), n.in(1).shape(), n.out(0).shape());
        IntArray a = (IntArray) n.in(0).storage().host();
        IntArray b = (IntArray) n.in(1).storage().host();
        IntArray o = (IntArray) n.out(0).storage().host();
        int r = meta[0];
        for (int i = 0; i < n.out(0).numel(); i++) {
            int rem = i;
            int ai = 0;
            int bi = 0;
            for (int d = r - 1; d >= 0; d--) {
                int c = rem % meta[1 + d];
                rem /= meta[1 + d];
                ai += c * meta[1 + r + d];
                bi += c * meta[1 + 2 * r + d];
            }
            o.set(i, BitwiseKernels.apply(n.iattr[0], a.get(ai), b.get(bi)));
        }
    }

    static final class Not implements Op {

        @Override
        public String name() {
            return "bitwise_not";
        }

        @Override
        public void emit(Emitter e, Node n) {
            IntArray x = e.i32(n.in(0));
            IntArray out = e.i32(n.out(0));
            int count = n.out(0).numel();
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, BitwiseKernels::complement, e.context(), x, out, count));
        }

        @Override
        public void reference(Node n) {
            IntArray x = (IntArray) n.in(0).storage().host();
            IntArray o = (IntArray) n.out(0).storage().host();
            for (int i = 0; i < x.getSize(); i++) {
                o.set(i, ~x.get(i));
            }
        }
    }
}
