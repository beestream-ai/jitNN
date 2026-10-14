package org.beehive.jitnn.engine;

import org.beehive.jitnn.Tensor;

/**
 * An operator: how to run it on a device (KernelContext kernels), how to run it on the host
 * (the reference used by tests), and how to differentiate it (in terms of other operators).
 */
public interface Op {

    /** A stable name; it is part of the compiled-trace cache key. */
    String name();

    /**
     * True when the operator takes FP16 tensors in half-precision inference
     * ({@code Autocast.enableHalf}); other operators get FP32 copies of FP16 inputs.
     */
    default boolean acceptsHalf() {
        return false;
    }

    /** Adds the device tasks for {@code node} to the graph being built. */
    void emit(Emitter e, Node node);

    /** Computes {@code node} on the host. Inputs are host-valid; output arrays are allocated. */
    void reference(Node node);

    /**
     * Gradients with respect to each input, given the gradient of the (first) output. Entries may
     * be null for inputs that need no gradient. Called with gradient tracking disabled.
     */
    default Tensor[] backward(Node node, Tensor gradOut) {
        throw new UnsupportedOperationException("no gradient for " + name());
    }
}
