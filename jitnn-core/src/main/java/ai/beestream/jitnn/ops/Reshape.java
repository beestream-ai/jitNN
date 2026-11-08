package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;

/** A zero-copy view with another shape. Never traced: it only exists for autograd. */
public final class Reshape implements Op {

    static final Reshape OP = new Reshape();

    private Reshape() {
    }

    public static Tensor apply(Tensor x, int... dims) {
        Shape target = resolve(x.numel(), dims);
        if (target.equals(x.shape())) {
            return x;
        }
        Tensor out = Tensor.alias(x, target);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { out }, null, null, false);
        return out;
    }

    static Shape resolve(int numel, int[] dims) {
        int[] d = dims.clone();
        int infer = -1;
        int known = 1;
        for (int i = 0; i < d.length; i++) {
            if (d[i] == -1) {
                if (infer >= 0) {
                    throw new IllegalArgumentException("only one dimension may be -1");
                }
                infer = i;
            } else {
                known *= d[i];
            }
        }
        if (infer >= 0) {
            d[infer] = known == 0 ? 0 : numel / known;
        }
        Shape s = Shape.of(d);
        if (s.numel() != numel) {
            throw new IllegalArgumentException("cannot reshape " + numel + " elements to " + s);
        }
        return s;
    }

    @Override
    public String name() {
        return "reshape";
    }

    @Override
    public void emit(Emitter e, Node node) {
        throw new IllegalStateException("reshape is a view and is never traced");
    }

    @Override
    public void reference(Node node) {
        throw new IllegalStateException("reshape is a view and is never traced");
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        return new Tensor[] { gy.reshape(n.in(0).shape().dims()) };
    }
}
