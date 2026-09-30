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

    /** A try-with-resources scope that restores the previous gradient mode. */
    public interface Scope extends AutoCloseable {
        @Override
        void close();
    }
}
