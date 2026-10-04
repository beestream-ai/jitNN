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
        SGD(0),
        SGD_MOMENTUM(1),
        ADAM(2);

        /** Optimiser state tensors per parameter. */
        final int states;

        Kind(int states) {
            this.states = states;
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
        int per = MultiTensorKernels.OPT_SLOTS;
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
        int slots = MultiTensorKernels.OPT_SLOTS;
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
