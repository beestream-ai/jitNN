package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.LowPrecision;

import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

/**
 * Conversion between FP32 and the half-precision types (round to nearest even, as PyTorch's
 * .half() and .bfloat16()), and between FP32 and INT32 (truncating toward zero, as a Java cast);
 * the others go through FP32. The gradient is cast back to the
 * input's type. iattr = [from, to] (DType ordinals).
 */
public final class Cast implements Op {

    static final Cast OP = new Cast();

    private Cast() {
    }

    public static Tensor apply(Tensor x, DType to) {
        DType from = x.dtype();
        if (from == to) {
            return x;
        }
        boolean integer = from == DType.I32 || to == DType.I32;
        DType other = from == DType.I32 ? to : from;
        if (integer && other != DType.F32 && other != DType.F16 && other != DType.BF16) {
            throw new IllegalArgumentException("INT32 converts to and from FP32, FP16 and BF16, got " + from + " to " + to);
        }
        if (!integer && (!from.isFloat() || !to.isFloat())) {
            throw new IllegalArgumentException("casts are between floating-point types and INT32, got " + from + " to " + to);
        }
        if (from != DType.F32 && to != DType.F32) {
            return apply(apply(x, DType.F32), to);
        }
        Tensor out = Tensor.result(x.shape(), to);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, new int[] { from.ordinal(), to.ordinal() }, null, true);
        return out;
    }

    /**
     * Records the cast of FP32 {@code x} into the existing tensor {@code out} (same shape; F16 or
     * BF16): for casts kept between steps in persistent storage, recomputed when x changes.
     */
    public static void into(Tensor x, Tensor out) {
        if (x.dtype() != DType.F32 || !(out.dtype() == DType.F16 || out.dtype() == DType.BF16) || x.numel() != out.numel()) {
            throw new IllegalArgumentException("cast into: needs FP32 into a half tensor of the same size, got " + x + " and " + out);
        }
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, new int[] { DType.F32.ordinal(), out.dtype().ordinal() }, null, true);
    }

    @Override
    public String name() {
        return "cast";
    }

    @Override
    public void emit(Emitter e, Node n) {
        DType from = DType.values()[n.iattr[0]];
        DType to = DType.values()[n.iattr[1]];
        int count = n.out(0).numel();
        if (to == DType.I32) {
            FloatArray x = e.f32(n.in(0));
            IntArray y = e.i32(n.out(0));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LowPrecision::f32ToI32, e.context(), x, y, count));
        } else if (from == DType.I32) {
            IntArray x = e.i32(n.in(0));
            FloatArray y = e.f32(n.out(0));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LowPrecision::i32ToF32, e.context(), x, y, count));
        } else if (from == DType.F32 && to == DType.F16) {
            FloatArray x = e.f32(n.in(0));
            HalfFloatArray y = e.f16(n.out(0));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LowPrecision::f32ToF16, e.context(), x, y, count));
        } else if (from == DType.F16) {
            HalfFloatArray x = e.f16(n.in(0));
            FloatArray y = e.f32(n.out(0));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LowPrecision::f16ToF32, e.context(), x, y, count));
        } else if (to == DType.BF16) {
            FloatArray x = e.f32(n.in(0));
            ShortArray y = e.bf16(n.out(0));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LowPrecision::f32ToBF16, e.context(), x, y, count));
        } else {
            ShortArray x = e.bf16(n.in(0));
            FloatArray y = e.f32(n.out(0));
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, LowPrecision::bf16ToF32, e.context(), x, y, count));
        }
    }

    @Override
    public void reference(Node n) {
        DType from = DType.values()[n.iattr[0]];
        DType to = DType.values()[n.iattr[1]];
        int count = n.out(0).numel();
        Object x = n.in(0).storage().host();
        Object y = n.out(0).storage().host();
        for (int i = 0; i < count; i++) {
            float v = switch (from) {
                case I32 -> ((IntArray) x).get(i);
                case F16 -> ((HalfFloatArray) x).get(i).getFloat32();
                case BF16 -> Float.intBitsToFloat(((int) ((ShortArray) x).get(i)) << 16);
                default -> ((FloatArray) x).get(i);
            };
            switch (to) {
                case I32 -> ((IntArray) y).set(i, (int) v);
                case F16 -> ((HalfFloatArray) y).set(i, new HalfFloat(Float.float16ToFloat(Float.floatToFloat16(v))));
                case BF16 -> ((ShortArray) y).set(i, bf16(v));
                default -> ((FloatArray) y).set(i, v);
            }
        }
    }

    /** Host FP32 to bfloat16 bits, round to nearest even (written independently of the kernel). */
    static short bf16(float v) {
        if (Float.isNaN(v)) {
            return (short) ((Float.floatToRawIntBits(v) >>> 16) | 0x40);
        }
        int bits = Float.floatToRawIntBits(v);
        int lsb = (bits >>> 16) & 1;
        return (short) ((bits + 0x7FFF + lsb) >>> 16);
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        if (n.in(0).dtype() == DType.I32 || n.out(0).dtype() == DType.I32) {
            return new Tensor[] { null };
        }
        return new Tensor[] { apply(gy, n.in(0).dtype()) };
    }
}
