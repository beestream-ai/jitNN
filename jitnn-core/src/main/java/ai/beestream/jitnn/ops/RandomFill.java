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
import ai.beestream.jitnn.kernels.RandomKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Philox4x32-10 random fill. iattr = [seedLo, seedHi, stream, mode]; fattr = [a, b]: uniform in
 * [a, b) for mode 0, normal with mean a and std b for mode 1.
 */
public final class RandomFill implements Op {

    static final RandomFill OP = new RandomFill();

    private RandomFill() {
    }

    /**
     * The same stream with the key taken from {@code seed}, an INT32 [2] tensor [lo, hi]: a replayed
     * plan draws new values from a new seed without recompiling.
     */
    public static Tensor apply(Shape shape, Tensor seed, int mode, float a, float b) {
        if (seed.dtype() != DType.I32 || seed.numel() != 2) {
            throw new IllegalArgumentException("random seed must be an INT32 tensor of 2 elements");
        }
        Tensor out = Tensor.result(shape, DType.F32);
        Tensor.record(OP, new Tensor[] { seed }, new Tensor[] { out }, new int[] { 0, 0, 0, mode }, new float[] { a, b }, true);
        return out;
    }

    public static Tensor apply(Shape shape, long seed, int mode, float a, float b) {
        Tensor out = Tensor.result(shape, DType.F32);
        Tensor.record(OP, new Tensor[0], new Tensor[] { out }, new int[] { (int) seed, (int) (seed >> 32), 0, mode }, new float[] { a, b }, true);
        return out;
    }

    @Override
    public String name() {
        return "philox";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray out = e.f32(n.out(0));
        int seedLo = n.iattr[0];
        int seedHi = n.iattr[1];
        int stream = n.iattr[2];
        int mode = n.iattr[3];
        float a = n.fattr[0];
        float b = n.fattr[1];
        int count = n.out(0).numel();
        if (n.inputs.length == 1) {
            IntArray seed = e.i32(n.in(0));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, RandomKernels::philoxSeeded, e.context(), out, seed, stream, mode, a, b, count, RandomKernels.ROUNDS));
            return;
        }
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, RandomKernels::philox, e.context(), out, seedLo, seedHi, stream, mode, a, b, count, RandomKernels.ROUNDS));
    }

    /** The host fill is already parallel over Philox's independent counter blocks: it is the CPU kernel too. */
    @Override
    public boolean cpu(Node n) {
        reference(n);
        return true;
    }

    @Override
    public void reference(Node n) {
        FloatArray out = (FloatArray) n.out(0).storage().host();
        int count = n.out(0).numel();
        int seedLo = n.iattr[0];
        int seedHi = n.iattr[1];
        if (n.inputs.length == 1) {
            IntArray seed = (IntArray) n.in(0).storage().host();
            seedLo = seed.get(0);
            seedHi = seed.get(1);
        }
        int lo = seedLo;
        int hi = seedHi;
        int blocks = (count + 3) / 4;
        // Philox is counter-based: blocks are independent, so large fills use every core (same values)
        java.util.stream.IntStream range = java.util.stream.IntStream.range(0, (blocks + 4095) / 4096);
        (blocks > 65536 ? range.parallel() : range).forEach(chunk -> {
            float[] block = new float[4];
            int end = Math.min(blocks, (chunk + 1) * 4096);
            for (int t = chunk * 4096; t < end; t++) {
                generate(t, lo, hi, n.iattr[2], n.iattr[3], n.fattr[0], n.fattr[1], block);
                for (int j = 0; j < 4 && t * 4 + j < count; j++) {
                    out.set(t * 4 + j, block[j]);
                }
            }
        });
    }

    /** Host Philox, the same bit-level algorithm as the kernel; transcendentals in double. */
    static void generate(int counter, int seedLo, int seedHi, int stream, int mode, float a, float b, float[] out) {
        int[] w = philox(counter, stream, 0, 0, seedLo, seedHi);
        if (mode == 0) {
            for (int j = 0; j < 4; j++) {
                out[j] = a + (b - a) * unit(w[j]);
            }
        } else {
            for (int j = 0; j < 4; j += 2) {
                double r = Math.sqrt(-2.0 * Math.log(1.0 - unit(w[j])));
                double th = 6.2831855f * unit(w[j + 1]);
                out[j] = (float) (a + b * r * Math.cos(th));
                out[j + 1] = (float) (a + b * r * Math.sin(th));
            }
        }
    }

    /** Philox4x32-10 on counter (c0..c3) and key (k0, k1); returns the four output words. */
    public static int[] philox(int c0, int c1, int c2, int c3, int k0, int k1) {
        for (int r = 0; r < 10; r++) {
            long p0 = 0xD2511F53L * (c0 & 0xFFFFFFFFL);
            long p1 = 0xCD9E8D57L * (c2 & 0xFFFFFFFFL);
            int hi0 = (int) (p0 >>> 32);
            int lo0 = (int) p0;
            int hi1 = (int) (p1 >>> 32);
            int lo1 = (int) p1;
            c0 = hi1 ^ c1 ^ k0;
            c1 = lo1;
            c2 = hi0 ^ c3 ^ k1;
            c3 = lo0;
            k0 += 0x9E3779B9;
            k1 += 0xBB67AE85;
        }
        return new int[] { c0, c1, c2, c3 };
    }

    static float unit(int w) {
        return (w >>> 8) * 5.9604645e-8f;
    }
}
