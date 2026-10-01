package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.RandomKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Inverted dropout with a counter-based mask: inputs [x, seed] where seed is an INT32 [2] tensor
 * created on the host each step (so a replayed plan draws a new mask without recompiling). The
 * gradient regenerates the same mask from the seed instead of storing it. iattr = [stream];
 * fattr = [p].
 */
public final class Dropout implements Op {

    static final Dropout OP = new Dropout();

    private Dropout() {
    }

    public static Tensor apply(Tensor x, float p, Tensor seed, int stream) {
        Checks.f32(x);
        if (seed.dtype() != DType.I32 || seed.numel() != 2) {
            throw new IllegalArgumentException("dropout seed must be an INT32 tensor of 2 elements");
        }
        if (p < 0 || p >= 1) {
            throw new IllegalArgumentException("dropout probability must be in [0, 1), got " + p);
        }
        Tensor y = Tensor.result(x.shape(), DType.F32);
        Tensor.record(OP, new Tensor[] { x, seed }, new Tensor[] { y }, new int[] { stream }, new float[] { p }, true);
        return y;
    }

    @Override
    public String name() {
        return "dropout";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        IntArray seed = e.i32(n.in(1));
        FloatArray y = e.f32(n.out(0));
        int stream = n.iattr[0];
        float p = n.fattr[0];
        float scale = 1.0f / (1.0f - p);
        int count = n.out(0).numel();
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, RandomKernels::dropout, e.context(), x, y, seed, stream, p, scale, count, RandomKernels.ROUNDS));
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        IntArray seed = (IntArray) n.in(1).storage().host();
        FloatArray y = (FloatArray) n.out(0).storage().host();
        float p = n.fattr[0];
        float scale = 1.0f / (1.0f - p);
        for (int i = 0; i < n.out(0).numel(); i++) {
            int w = RandomFill.philox(i >> 2, n.iattr[0], 0, 0, seed.get(0), seed.get(1))[i & 3];
            y.set(i, RandomFill.unit(w) >= p ? x.get(i) * scale : 0.0f);
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        return new Tensor[] { apply(gy, n.fattr[0], n.in(1), n.iattr[0]), null };
    }
}
