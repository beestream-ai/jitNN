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

    /** A try-with-resources scope that restores the previous gradient mode. */
    public interface Scope extends AutoCloseable {
        @Override
        void close();
    }
}
