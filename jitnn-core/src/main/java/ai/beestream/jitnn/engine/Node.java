package ai.beestream.jitnn.engine;

import java.util.Arrays;

import ai.beestream.jitnn.Tensor;

/**
 * One operator application. Traced nodes wait in the thread's {@link Trace} until a sync point
 * runs them; untraced nodes (views) exist only so autograd can differentiate through them.
 */
public final class Node {

    /** The operator applied. */
    public final Op op;
    /** The operator's inputs. */
    public final Tensor[] inputs;
    /** The operator's outputs. */
    public final Tensor[] outputs;
    /** The integer attributes of this application (shapes, axes, flags); never null. */
    public final int[] iattr;
    /** The float attributes of this application (scales, constants); never null. */
    public final float[] fattr;
    final boolean traced;
    boolean executed;

    /**
     * An application of {@code op}; a null attribute array becomes an empty one. Untraced nodes (views)
     * never run, they exist for autograd. Operators create nodes through {@link Tensor#record}.
     */
    public Node(Op op, Tensor[] inputs, Tensor[] outputs, int[] iattr, float[] fattr, boolean traced) {
        this.op = op;
        this.inputs = inputs;
        this.outputs = outputs;
        this.iattr = iattr == null ? new int[0] : iattr;
        this.fattr = fattr == null ? new float[0] : fattr;
        this.traced = traced;
    }

    /** Whether the node has run (always true for an untraced node). */
    public boolean isExecuted() {
        return executed || !traced;
    }

    /** Input {@code i}. */
    public Tensor in(int i) {
        return inputs[i];
    }

    /** Output {@code i}. */
    public Tensor out(int i) {
        return outputs[i];
    }

    @Override
    public String toString() {
        return op.name() + Arrays.toString(iattr) + Arrays.toString(fattr);
    }
}
