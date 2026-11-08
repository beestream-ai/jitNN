package ai.beestream.jitnn.nn;

import ai.beestream.jitnn.Tensor;

/** Residual connection: relu(main(x) + shortcut(x)) or main(x) + shortcut(x). */
public final class Add extends Module {

    private final Module main;
    private final Module shortcut;
    private final boolean relu;

    /** {@code shortcut} null means identity. */
    public Add(Module main, Module shortcut, boolean relu) {
        this.main = child(main);
        this.shortcut = shortcut == null ? null : child(shortcut);
        this.relu = relu;
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor y = main.forward(x).add(shortcut == null ? x : shortcut.forward(x));
        return relu ? y.relu() : y;
    }
}
