package org.beehive.jitnn.ops;

import java.util.List;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.MultiTensorKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * In-place optimiser updates of up to {@link MultiTensorKernels#OPT_SLOTS} persistent tensors in
 * one launch (as PyTorch's foreach kernels). The hyper-parameter tensor is a fresh host input each
 * step, so learning-rate schedules and Adam's bias correction work under plan replay. An optional
 * {@code stats} tensor [norm, factor, skip] (see GradStats) scales every gradient by the factor and
 * skips the update when skip is non-zero. Inputs: per tensor [param, grad, state...], then hyper,
 * then the optional stats; iattr = [kind, hasStats, count].
 */
public final class OptimStep implements Op {

    public enum Kind {
        SGD(0, 4),
        SGD_MOMENTUM(1, 4),
        ADAM(2, 4),
        AMSGRAD(3, 3),
        ADAGRAD(1, 4),
        RMSPROP(3, 3),
        ADADELTA(2, 4),
        ADAMAX(2, 4),
        NADAM(2, 4);

        /** Optimiser state tensors per parameter. */
        final int states;
        /** Parameters per launch (TornadoVM tasks take at most 20 arguments). */
        final int slots;

        Kind(int states, int slots) {
            this.states = states;
            this.slots = slots;
        }
    }

    static final OptimStep OP = new OptimStep();

    private OptimStep() {
    }

    /**
     * Records the updates of {@code params} with their gradients and per-parameter state
     * ({@code states.get(s).get(i)} is state s of parameter i: none for SGD, the momentum buffer,
     * or Adam's m and v), in launches of up to {@link MultiTensorKernels#OPT_SLOTS}. hyper:
     * SGD [lr, momentum, weightDecay, nesterov]; Adam [lr, beta1, beta2, eps, 1 - beta1^t,
     * 1 - beta2^t, weightDecay, decoupled].
     */
    public static void apply(Kind kind, List<Tensor> params, List<Tensor> grads, List<List<Tensor>> states, Tensor hyper, Tensor stats) {
        int per = kind.slots;
        for (int start = 0; start < params.size(); start += per) {
            int count = Math.min(per, params.size() - start);
            List<Tensor> in = new java.util.ArrayList<>();
            List<Tensor> out = new java.util.ArrayList<>();
            for (int i = start; i < start + count; i++) {
                Tensor p = params.get(i);
                Tensor g = grads.get(i);
                check(p, g);
                in.add(p);
                in.add(g);
                out.add(p);
                for (List<Tensor> state : states) {
                    check(state.get(i), g);
                    in.add(state.get(i));
                    out.add(state.get(i));
                }
            }
            in.add(hyper);
            if (stats != null) {
                in.add(stats);
            }
            Tensor.record(OP, in.toArray(new Tensor[0]), out.toArray(new Tensor[0]), new int[] { kind.ordinal(), stats != null ? 1 : 0, count }, null, true);
        }
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
        Kind kind = Kind.values()[n.iattr[0]];
        int count = n.iattr[2];
        int width = 2 + kind.states;
        int slots = kind.slots;
        // [slot][p, g, state...]; unused slots get their own one-element arrays and size 0
        FloatArray[][] a = new FloatArray[slots][width];
        int[] sizes = new int[slots];
        int total = 0;
        for (int k = 0; k < slots; k++) {
            for (int j = 0; j < width; j++) {
                a[k][j] = k < count ? e.f32(n.in(k * width + j)) : e.scratch(1);
            }
            sizes[k] = k < count ? n.in(k * width).numel() : 0;
            total += sizes[k];
        }
        FloatArray h = e.f32(n.in(count * width));
        FloatArray stats = n.iattr[1] != 0 ? e.f32(n.in(count * width + 1)) : e.constantF32(0.0f, 1.0f, 0.0f);
        IntArray sz = e.constant(sizes);
        switch (kind) {
            case SGD -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::sgd, e.context(), a[0][0], a[0][1], a[1][0], a[1][1], a[2][0], a[2][1],
                    a[3][0], a[3][1], h, stats, sz));
            case SGD_MOMENTUM -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::sgdMomentum, e.context(), a[0][0], a[0][1], a[0][2], a[1][0], a[1][1],
                    a[1][2], a[2][0], a[2][1], a[2][2], a[3][0], a[3][1], a[3][2], h, stats, sz));
            case ADAM -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::adam, e.context(), a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1],
                    a[1][2], a[1][3], a[2][0], a[2][1], a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3], h, stats, sz));
            case AMSGRAD -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::amsgrad, e.context(), a[0][0], a[0][1], a[0][2], a[0][3], a[0][4],
                    a[1][0], a[1][1], a[1][2], a[1][3], a[1][4], a[2][0], a[2][1], a[2][2], a[2][3], a[2][4], h, stats, sz));
            case ADAGRAD -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::adagrad, e.context(), a[0][0], a[0][1], a[0][2], a[1][0], a[1][1],
                    a[1][2], a[2][0], a[2][1], a[2][2], a[3][0], a[3][1], a[3][2], h, stats, sz));
            case RMSPROP -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::rmsprop, e.context(), a[0][0], a[0][1], a[0][2], a[0][3], a[0][4],
                    a[1][0], a[1][1], a[1][2], a[1][3], a[1][4], a[2][0], a[2][1], a[2][2], a[2][3], a[2][4], h, stats, sz));
            case ADADELTA -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::adadelta, e.context(), a[0][0], a[0][1], a[0][2], a[0][3], a[1][0],
                    a[1][1], a[1][2], a[1][3], a[2][0], a[2][1], a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3], h, stats, sz));
            case ADAMAX -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::adamax, e.context(), a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1],
                    a[1][2], a[1][3], a[2][0], a[2][1], a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3], h, stats, sz));
            case NADAM -> e.task(Emitter.grid1D(total), (tg, id) -> tg.task(id, MultiTensorKernels::nadam, e.context(), a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1],
                    a[1][2], a[1][3], a[2][0], a[2][1], a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3], h, stats, sz));
        }
    }

    /** One element of the newer optimisers in double precision (states updated in place); returns the new parameter. */
    private static double referenceStep(Kind kind, FloatArray h, double w, double grad, FloatArray[] st, int i) {
        switch (kind) {
            case AMSGRAD -> {
                double lr = h.get(0), b1 = h.get(1), b2 = h.get(2), wd = h.get(6);
                if (h.get(7) != 0) {
                    w -= lr * wd * w;
                } else {
                    grad += wd * w;
                }
                double m = b1 * st[0].get(i) + (1 - b1) * grad;
                double v = b2 * st[1].get(i) + (1 - b2) * grad * grad;
                double vm = Math.max(st[2].get(i), v);
                st[0].set(i, (float) m);
                st[1].set(i, (float) v);
                st[2].set(i, (float) vm);
                return w - lr * (m / h.get(4)) / (Math.sqrt(vm) / Math.sqrt(h.get(5)) + h.get(3));
            }
            case ADAGRAD -> {
                grad += h.get(2) * w;
                double sum = st[0].get(i) + grad * grad;
                st[0].set(i, (float) sum);
                return w - h.get(0) * grad / (Math.sqrt(sum) + h.get(1));
            }
            case RMSPROP -> {
                double alpha = h.get(1);
                grad += h.get(3) * w;
                double sq = alpha * st[0].get(i) + (1 - alpha) * grad * grad;
                st[0].set(i, (float) sq);
                double avg;
                if (h.get(5) != 0) {
                    double ga = alpha * st[2].get(i) + (1 - alpha) * grad;
                    st[2].set(i, (float) ga);
                    avg = Math.sqrt(sq - ga * ga) + h.get(2);
                } else {
                    avg = Math.sqrt(sq) + h.get(2);
                }
                if (h.get(4) != 0) {
                    double b = h.get(4) * st[1].get(i) + grad / avg;
                    st[1].set(i, (float) b);
                    return w - h.get(0) * b;
                }
                return w - h.get(0) * grad / avg;
            }
            case ADADELTA -> {
                double rho = h.get(1), eps = h.get(2);
                grad += h.get(3) * w;
                double sq = rho * st[0].get(i) + (1 - rho) * grad * grad;
                double delta = Math.sqrt(st[1].get(i) + eps) / Math.sqrt(sq + eps) * grad;
                st[0].set(i, (float) sq);
                st[1].set(i, (float) (rho * st[1].get(i) + (1 - rho) * delta * delta));
                return w - h.get(0) * delta;
            }
            case ADAMAX -> {
                double b1 = h.get(1);
                grad += h.get(5) * w;
                double m = b1 * st[0].get(i) + (1 - b1) * grad;
                double u = Math.max(h.get(2) * st[1].get(i), Math.abs(grad) + h.get(3));
                st[0].set(i, (float) m);
                st[1].set(i, (float) u);
                return w - h.get(0) / h.get(4) * m / u;
            }
            case NADAM -> {
                double b1 = h.get(1), b2 = h.get(2);
                grad += h.get(4) * w;
                double m = b1 * st[0].get(i) + (1 - b1) * grad;
                double v = b2 * st[1].get(i) + (1 - b2) * grad * grad;
                st[0].set(i, (float) m);
                st[1].set(i, (float) v);
                double denom = Math.sqrt(v / h.get(8)) + h.get(3);
                return w - h.get(9) * grad / denom - h.get(10) * m / denom;
            }
            default -> throw new IllegalStateException(kind.toString());
        }
    }

    @Override
    public void reference(Node n) {
        Kind kind = Kind.values()[n.iattr[0]];
        int count = n.iattr[2];
        int width = 2 + kind.states;
        FloatArray h = (FloatArray) n.in(count * width).storage().host();
        double factor = 1.0;
        if (n.iattr[1] != 0) {
            FloatArray stats = (FloatArray) n.in(count * width + 1).storage().host();
            if (stats.get(2) != 0.0f) {
                return;
            }
            factor = stats.get(1);
        }
        for (int k = 0; k < count; k++) {
            FloatArray p = (FloatArray) n.in(k * width).storage().host();
            FloatArray g = (FloatArray) n.in(k * width + 1).storage().host();
            int size = n.in(k * width).numel();
            switch (kind) {
                case SGD -> {
                    for (int i = 0; i < size; i++) {
                        double w = p.get(i);
                        p.set(i, (float) (w - h.get(0) * (g.get(i) * factor + h.get(2) * w)));
                    }
                }
                case SGD_MOMENTUM -> {
                    FloatArray buf = (FloatArray) n.in(k * width + 2).storage().host();
                    for (int i = 0; i < size; i++) {
                        double w = p.get(i);
                        double grad = g.get(i) * factor + h.get(2) * w;
                        double v = h.get(1) * buf.get(i) + grad;
                        buf.set(i, (float) v);
                        double step = h.get(3) != 0 ? grad + h.get(1) * v : v;
                        p.set(i, (float) (w - h.get(0) * step));
                    }
                }
                case AMSGRAD, ADAGRAD, RMSPROP, ADADELTA, ADAMAX, NADAM -> {
                    FloatArray[] st = new FloatArray[kind.states];
                    for (int j = 0; j < st.length; j++) {
                        st[j] = (FloatArray) n.in(k * width + 2 + j).storage().host();
                    }
                    for (int i = 0; i < size; i++) {
                        p.set(i, (float) referenceStep(kind, h, p.get(i), g.get(i) * factor, st, i));
                    }
                }
                case ADAM -> {
                    FloatArray m = (FloatArray) n.in(k * width + 2).storage().host();
                    FloatArray v = (FloatArray) n.in(k * width + 3).storage().host();
                    double lr = h.get(0);
                    double b1 = h.get(1);
                    double b2 = h.get(2);
                    double wd = h.get(6);
                    for (int i = 0; i < size; i++) {
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
}
