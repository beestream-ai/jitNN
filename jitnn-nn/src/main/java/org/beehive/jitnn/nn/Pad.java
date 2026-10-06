package org.beehive.jitnn.nn;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;

/** Padding of the spatial axes: constant, reflect or replicate (PyTorch's nn.ConstantPadNd, ReflectionPadNd, ReplicationPadNd). */
public final class Pad extends Module {

    private final int[] before;
    private final int[] after;
    private final F.PadMode mode;
    private final float value;

    public Pad(int[] before, int[] after, F.PadMode mode, float value) {
        this.before = before.clone();
        this.after = after.clone();
        this.mode = mode;
        this.value = value;
    }

    @Override
    public Tensor forward(Tensor x) {
        return F.pad(x, before, after, mode, value);
    }
}
