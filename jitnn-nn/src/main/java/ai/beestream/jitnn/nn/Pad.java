package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;

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
