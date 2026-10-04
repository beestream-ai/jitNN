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
 * An optional {@code stats} tensor [norm, factor, skip] (last input, iattr[1] = 1, see GradStats)
 * scales every gradient by the factor and skips the update when skip is non-zero.
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
    public static void sgd(Tensor param, Tensor grad, Tensor hyper, Tensor stats) {
        check(param, grad);
        record(new Tensor[] { param, grad, hyper }, new Tensor[] { param }, Kind.SGD, stats);
    }

    /** SGD with momentum buffer {@code buf}, same hyper layout as {@link #sgd}. */
    public static void sgdMomentum(Tensor param, Tensor grad, Tensor buf, Tensor hyper, Tensor stats) {
        check(param, grad);
        check(buf, grad);
        record(new Tensor[] { param, grad, buf, hyper }, new Tensor[] { param, buf }, Kind.SGD_MOMENTUM, stats);
    }

    /** Adam/AdamW; hyper = [lr, beta1, beta2, eps, 1 - beta1^t, 1 - beta2^t, weightDecay, decoupled]. */
    public static void adam(Tensor param, Tensor grad, Tensor m, Tensor v, Tensor hyper, Tensor stats) {
        check(param, grad);
        check(m, grad);
        check(v, grad);
        record(new Tensor[] { param, grad, m, v, hyper }, new Tensor[] { param, m, v }, Kind.ADAM, stats);
    }

    private static void record(Tensor[] in, Tensor[] out, Kind kind, Tensor stats) {
        Tensor[] inputs = in;
        if (stats != null) {
            inputs = java.util.Arrays.copyOf(in, in.length + 1);
            inputs[in.length] = stats;
        }
        Tensor.record(OP, inputs, out, new int[] { kind.ordinal(), stats != null ? 1 : 0 }, null, true);
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
        FloatArray stats = n.iattr[1] != 0 ? e.f32(n.in(n.inputs.length - 1)) : e.constantF32(0.0f, 1.0f, 0.0f);
        switch (Kind.values()[n.iattr[0]]) {
            case SGD -> {
                FloatArray h = e.f32(n.in(2));
                e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, OptimKernels::sgd, e.context(), p, g, h, stats, count));
            }
            case SGD_MOMENTUM -> {
                FloatArray buf = e.f32(n.in(2));
                FloatArray h = e.f32(n.in(3));
                e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, OptimKernels::sgdMomentum, e.context(), p, g, buf, h, stats, count));
            }
            case ADAM -> {
                FloatArray m = e.f32(n.in(2));
                FloatArray v = e.f32(n.in(3));
                FloatArray h = e.f32(n.in(4));
                e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, OptimKernels::adam, e.context(), p, g, m, v, h, stats, count));
            }
        }
    }

    @Override
    public void reference(Node n) {
        FloatArray p = (FloatArray) n.in(0).storage().host();
        FloatArray g = (FloatArray) n.in(1).storage().host();
        int count = n.in(0).numel();
        double factor = 1.0;
        if (n.iattr[1] != 0) {
            FloatArray stats = (FloatArray) n.in(n.inputs.length - 1).storage().host();
            if (stats.get(2) != 0.0f) {
                return;
            }
            factor = stats.get(1);
        }
        switch (Kind.values()[n.iattr[0]]) {
            case SGD -> {
                FloatArray h = (FloatArray) n.in(2).storage().host();
                for (int i = 0; i < count; i++) {
                    double w = p.get(i);
                    p.set(i, (float) (w - h.get(0) * (g.get(i) * factor + h.get(2) * w)));
                }
            }
            case SGD_MOMENTUM -> {
                FloatArray buf = (FloatArray) n.in(2).storage().host();
                FloatArray h = (FloatArray) n.in(3).storage().host();
                for (int i = 0; i < count; i++) {
                    double w = p.get(i);
                    double grad = g.get(i) * factor + h.get(2) * w;
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
                    double grad = g.get(i) * factor;
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
