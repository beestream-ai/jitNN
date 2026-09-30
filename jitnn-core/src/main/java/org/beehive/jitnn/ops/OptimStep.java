package org.beehive.jitnn.ops;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.OptimKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * In-place optimiser updates on persistent tensors. The hyper-parameter tensor is a fresh host
 * input each step, so learning-rate schedules and Adam's bias correction work under plan replay.
 */
public final class OptimStep implements Op {

    public enum Kind {
        SGD,
        SGD_MOMENTUM,
        ADAM
    }

    static final OptimStep OP = new OptimStep();

    private OptimStep() {
    }

    /** SGD: inputs [param, grad, hyper]; hyper = [lr, momentum, weightDecay, nesterov]. */
    public static void sgd(Tensor param, Tensor grad, Tensor hyper) {
        check(param, grad);
        Tensor.record(OP, new Tensor[] { param, grad, hyper }, new Tensor[] { param }, new int[] { Kind.SGD.ordinal() }, null, true);
    }

    /** SGD with momentum buffer {@code buf}, same hyper layout as {@link #sgd}. */
    public static void sgdMomentum(Tensor param, Tensor grad, Tensor buf, Tensor hyper) {
        check(param, grad);
        check(buf, grad);
        Tensor.record(OP, new Tensor[] { param, grad, buf, hyper }, new Tensor[] { param, buf }, new int[] { Kind.SGD_MOMENTUM.ordinal() }, null, true);
    }

    /** Adam/AdamW; hyper = [lr, beta1, beta2, eps, 1 - beta1^t, 1 - beta2^t, weightDecay, decoupled]. */
    public static void adam(Tensor param, Tensor grad, Tensor m, Tensor v, Tensor hyper) {
        check(param, grad);
        check(m, grad);
        check(v, grad);
        Tensor.record(OP, new Tensor[] { param, grad, m, v, hyper }, new Tensor[] { param, m, v }, new int[] { Kind.ADAM.ordinal() }, null, true);
    }

    private static void check(Tensor state, Tensor grad) {
        if (!state.storage().isPersistent()) {
            throw new IllegalArgumentException("optimiser state must be persistent: " + state);
        }
        if (!state.shape().equals(grad.shape())) {
            throw new IllegalArgumentException("shape mismatch: " + state.shape() + " vs " + grad.shape());
        }
    }

    @Override
    public String name() {
        return "optim";
    }

    @Override
    public void emit(Emitter e, Node n) {
        int count = n.in(0).numel();
        FloatArray p = e.f32(n.in(0));
        FloatArray g = e.f32(n.in(1));
        switch (Kind.values()[n.iattr[0]]) {
            case SGD -> {
                FloatArray h = e.f32(n.in(2));
                e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, OptimKernels::sgd, e.context(), p, g, h, count));
            }
            case SGD_MOMENTUM -> {
                FloatArray buf = e.f32(n.in(2));
                FloatArray h = e.f32(n.in(3));
                e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, OptimKernels::sgdMomentum, e.context(), p, g, buf, h, count));
            }
            case ADAM -> {
                FloatArray m = e.f32(n.in(2));
                FloatArray v = e.f32(n.in(3));
                FloatArray h = e.f32(n.in(4));
                e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, OptimKernels::adam, e.context(), p, g, m, v, h, count));
            }
        }
    }

    @Override
    public void reference(Node n) {
        FloatArray p = (FloatArray) n.in(0).storage().host();
        FloatArray g = (FloatArray) n.in(1).storage().host();
        int count = n.in(0).numel();
        switch (Kind.values()[n.iattr[0]]) {
            case SGD -> {
                FloatArray h = (FloatArray) n.in(2).storage().host();
                for (int i = 0; i < count; i++) {
                    double w = p.get(i);
                    p.set(i, (float) (w - h.get(0) * (g.get(i) + h.get(2) * w)));
                }
            }
            case SGD_MOMENTUM -> {
                FloatArray buf = (FloatArray) n.in(2).storage().host();
                FloatArray h = (FloatArray) n.in(3).storage().host();
                for (int i = 0; i < count; i++) {
                    double w = p.get(i);
                    double grad = g.get(i) + h.get(2) * w;
                    double v = h.get(1) * buf.get(i) + grad;
                    buf.set(i, (float) v);
                    double step = h.get(3) != 0 ? grad + h.get(1) * v : v;
                    p.set(i, (float) (w - h.get(0) * step));
                }
            }
            case ADAM -> {
                FloatArray m = (FloatArray) n.in(2).storage().host();
                FloatArray v = (FloatArray) n.in(3).storage().host();
                FloatArray h = (FloatArray) n.in(4).storage().host();
                double lr = h.get(0);
                double b1 = h.get(1);
                double b2 = h.get(2);
                double wd = h.get(6);
                for (int i = 0; i < count; i++) {
                    double w = p.get(i);
                    double grad = g.get(i);
                    if (h.get(7) != 0) {
                        w -= lr * wd * w;
                    } else {
                        grad += wd * w;
                    }
                    double mi = b1 * m.get(i) + (1 - b1) * grad;
                    double vi = b2 * v.get(i) + (1 - b2) * grad * grad;
                    m.set(i, (float) mi);
                    v.set(i, (float) vi);
                    p.set(i, (float) (w - lr * (mi / h.get(4)) / (Math.sqrt(vi / h.get(5)) + h.get(3))));
                }
            }
        }
    }
}
