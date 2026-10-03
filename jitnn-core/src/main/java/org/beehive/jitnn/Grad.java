package org.beehive.jitnn;

/** Whether new operators record what autograd needs. On by default, per thread. */
public final class Grad {

    private static final ThreadLocal<Boolean> ENABLED = ThreadLocal.withInitial(() -> Boolean.TRUE);

    private Grad() {
    }

    public static boolean isEnabled() {
        return ENABLED.get();
    }

    /** Disables gradient recording until the returned scope is closed. */
    public static Scope disabled() {
        boolean previous = ENABLED.get();
        ENABLED.set(Boolean.FALSE);
        return () -> ENABLED.set(previous);
    }

    /**
     * Runs a forward pass for inference: no gradient recording, and every tensor created inside
     * except the returned one is internal, so its buffer is reused as soon as the computation no
     * longer needs it. Reading those intermediates after the call is not supported.
     */
    public static Tensor inference(java.util.function.Supplier<Tensor> forward) {
        Tensor out;
        try (Scope ignored = disabled(); AutoCloseable internal = org.beehive.jitnn.engine.Storage.internalScope()) {
            out = forward.get();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        out.storage().markVisible();
        return out;
    }

    /**
     * Rescales the gradients of {@code params} so that their global L2 norm is at most
     * {@code maxNorm}, as PyTorch's {@code clip_grad_norm_}: every gradient is multiplied by
     * min(1, maxNorm / (norm + 1e-6)). Runs lazily on the device with the rest of the step and
     * returns the norm before clipping as a scalar tensor (reading it is a sync point).
     */
    public static Tensor clipNorm(java.util.List<Tensor> params, float maxNorm) {
        java.util.List<Tensor> squares = new java.util.ArrayList<>();
        try (Scope ignored = disabled()) {
            for (Tensor p : params) {
                if (p.grad() != null) {
                    squares.add(p.grad().square().sum().reshape(1));
                }
            }
            if (squares.isEmpty()) {
                return Tensor.scalar(0.0f);
            }
            Tensor norm = Tensor.cat(0, squares.toArray(new Tensor[0])).sum().sqrt();
            Tensor coef = norm.add(1e-6f).reciprocal().mul(maxNorm).clamp(0.0f, 1.0f);
            for (Tensor p : params) {
                if (p.grad() != null) {
                    p.setGrad(p.grad().mul(coef));
                }
            }
            return norm;
        }
    }

    /** A try-with-resources scope that restores the previous gradient mode. */
    public interface Scope extends AutoCloseable {
        @Override
        void close();
    }
}
