package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.LowPrecision;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** A one-element FP32 flag: 1 when any value of x is NaN or infinite (tested on the bit pattern), else 0. */
public final class NonFinite implements Op {

    static final NonFinite OP = new NonFinite();

    private NonFinite() {
    }

    public static Tensor apply(Tensor x) {
        Checks.f32(x);
        Tensor flag = Tensor.result(Shape.of(1), DType.F32);
        Tensor.record(OP, new Tensor[] { x }, new Tensor[] { flag }, null, null, true);
        return flag;
    }

    @Override
    public String name() {
        return "non_finite";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray flag = e.f32(n.out(0));
        int count = n.in(0).numel();
        e.task(Emitter.groups(1), (g, id) -> g.task(id, LowPrecision::anyNonFinite, e.context(), x, flag, count));
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        boolean bad = false;
        for (int i = 0; i < n.in(0).numel(); i++) {
            bad |= !Float.isFinite(x.get(i));
        }
        ((FloatArray) n.out(0).storage().host()).set(0, bad ? 1.0f : 0.0f);
    }
}
