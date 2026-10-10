package org.beehive.jitnn;

/**
 * Mixed precision, per thread: inside {@link #enable(DType)}, {@link F#linear}, {@link Tensor#matmul}
 * and the products of {@link F#scaledDotProductAttention} cast their operands to FP16 or BF16
 * and run the half-precision GEMMs, accumulating in FP32. Their results stay FP32 (PyTorch's
 * autocast rounds them to the half type), and every other operator runs in FP32. Gradients of
 * the products are computed in the half type and flow back to the FP32 parameters through the casts.
 * For inference, weights are cast once per weight version ({@link #weight}), and on CUDA a
 * Linear whose shape the tensor-core kernel covers reads its FP32 input directly, rounding it to
 * FP16 as the tiles are staged, so no separate cast runs.
 */
public final class Autocast {

    private static final ThreadLocal<DType> TYPE = new ThreadLocal<>();

    private Autocast() {
    }

    /** Enables autocast to {@code dtype} (F16 or BF16) until the returned scope is closed. */
    public static Grad.Scope enable(DType dtype) {
        if (dtype != DType.F16 && dtype != DType.BF16) {
            throw new IllegalArgumentException("autocast type must be F16 or BF16, got " + dtype);
        }
        DType previous = TYPE.get();
        TYPE.set(dtype);
        return () -> TYPE.set(previous);
    }

    /** The active autocast type, or null. */
    public static DType current() {
        return TYPE.get();
    }

    /** {@code t} cast to the autocast type when autocast is on and t is FP32; otherwise t. */
    static Tensor operand(Tensor t) {
        DType d = TYPE.get();
        return d != null && t.dtype() == DType.F32 ? t.to(d) : t;
    }

    private record Cast(long version, DType type, Tensor half) {
    }

    private static final java.util.Map<Tensor, Cast> WEIGHTS = java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /**
     * A weight cast to the autocast type, kept between steps while gradients are off: cast again
     * only when the weight's storage has changed since (its version), so inference does not
     * re-cast every weight on every step. While gradients are recorded the cast is part of the
     * graph, as {@link #operand}.
     */
    static Tensor weight(Tensor w) {
        DType d = TYPE.get();
        if (d == null || w.dtype() != DType.F32 || Grad.isEnabled()) {
            return operand(w);
        }
        long version = w.storage().version();
        Cast c = WEIGHTS.get(w);
        if (c == null || c.version() != version || c.type() != d) {
            c = new Cast(version, d, w.to(d));
            WEIGHTS.put(w, c);
        }
        return c.half();
    }
}
