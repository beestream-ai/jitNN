package ai.beestream.jitnn.engine;

import java.util.Arrays;

import ai.beestream.jitnn.Tensor;

/**
 * One operator application. Traced nodes wait in the thread's {@link Trace} until a sync point
 * runs them; untraced nodes (views) exist only so autograd can differentiate through them.
 */
public final class Node {

    public final Op op;
    public final Tensor[] inputs;
    public final Tensor[] outputs;
    public final int[] iattr;
    public final float[] fattr;
    final boolean traced;
    boolean executed;

    public Node(Op op, Tensor[] inputs, Tensor[] outputs, int[] iattr, float[] fattr, boolean traced) {
        this.op = op;
        this.inputs = inputs;
        this.outputs = outputs;
        this.iattr = iattr == null ? new int[0] : iattr;
        this.fattr = fattr == null ? new float[0] : fattr;
        this.traced = traced;
    }

    public boolean isExecuted() {
        return executed || !traced;
    }

    public Tensor in(int i) {
        return inputs[i];
    }

    public Tensor out(int i) {
        return outputs[i];
    }

    @Override
    public String toString() {
        return op.name() + Arrays.toString(iattr) + Arrays.toString(fattr);
    }
}
