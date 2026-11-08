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
package ai.beestream.jitnn;

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
    private static final ThreadLocal<Boolean> HALF = ThreadLocal.withInitial(() -> Boolean.FALSE);

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

    /**
     * Half-precision activations, as a model converted to FP16: inside the scope, linear layers,
     * element-wise operators, LayerNorm and RMSNorm and attention take and return FP16 tensors
     * (computing in FP32 inside), so activations stay FP16 between them, with half the memory
     * traffic. Operators without an FP16 form take FP32 copies of FP16 inputs and return FP32.
     * In inference weights are cast to FP16 once per weight version, as with {@link #enable}.
     *
     * <p>In training the weights stay FP32 (the master copy the optimiser updates) and are cast to
     * FP16 inside the graph; the backward pass runs the same way (autograd switches it on when the
     * graph holds FP16 tensors): parameters get gradients of their own type (FP32), and activations'
     * gradients stay as the operators compute them (mostly FP32). FP16 training needs loss scaling
     * ({@code GradScaler}).
     */
    public static Grad.Scope enableHalf() {
        DType previousType = TYPE.get();
        boolean previousHalf = HALF.get();
        TYPE.set(DType.F16);
        HALF.set(Boolean.TRUE);
        return () -> {
            TYPE.set(previousType);
            HALF.set(previousHalf);
        };
    }

    private static final ThreadLocal<Boolean> BACKWARD = ThreadLocal.withInitial(() -> Boolean.FALSE);

    /** Marks the backward pass (autograd recording gradient operators) until the scope is closed. */
    static Grad.Scope backward() {
        boolean previous = BACKWARD.get();
        BACKWARD.set(Boolean.TRUE);
        return () -> BACKWARD.set(previous);
    }

    /**
     * True when FP16 activations should stay FP16 through operators that compute in FP32 (inside
     * {@link #enableHalf()}, outside the backward pass): their results are cast back to FP16, so an
     * FP32 fallback does not turn the rest of the forward pass FP32.
     */
    public static boolean keepHalf() {
        return HALF.get() && !BACKWARD.get();
    }

    /** True inside {@link #enableHalf()}: FP16-capable operators return FP16. */
    public static boolean halfActivations() {
        return HALF.get();
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
     * A persistent FP32 weight cast to FP16, kept between steps while gradients are off in
     * persistent FP16 storage: cast again only when the weight's storage has changed since (its
     * version), so inference does not re-cast every weight on every step. While gradients are recorded the cast is part of the
     * graph, as {@link #operand}.
     */
    static Tensor weight(Tensor w) {
        DType d = TYPE.get();
        if (d != DType.F16 || w.dtype() != DType.F32 || Grad.isEnabled() || !w.storage().isPersistent()) {
            return operand(w);
        }
        long version = w.storage().version();
        Cast c = WEIGHTS.get(w);
        if (c == null || c.version() != version) {
            // the half copy lives in persistent storage (an intermediate's buffer would be reused by
            // the next execution), rewritten inside the trace that first sees the new weight
            Tensor half = c != null ? c.half() : Tensor.persistentHalf(new float[w.numel()], w.shape().dims());
            ai.beestream.jitnn.ops.Cast.into(w, half);
            c = new Cast(version, d, half);
            WEIGHTS.put(w, c);
        }
        return c.half();
    }
}
