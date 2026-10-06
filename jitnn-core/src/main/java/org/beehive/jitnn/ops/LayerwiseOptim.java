package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.LayerwiseOptimKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** In-place LAMB and LARS updates of one persistent parameter (see {@link LayerwiseOptimKernels}). */
public final class LayerwiseOptim {

    static final LambMoments LAMB_MOMENTS = new LambMoments();
    static final TrustUpdate TRUST_UPDATE = new TrustUpdate();
    static final LarsUpdate LARS_UPDATE = new LarsUpdate();

    private LayerwiseOptim() {
    }

    private static Tensor norm(Tensor x) {
        return x.square().sum().sqrt().reshape(1);
    }

    private static Tensor stats(Tensor stats) {
        return stats != null ? stats : Tensor.of(new float[] { 0, 1, 0 }, 3);
    }

    /** One LAMB step of p with gradient g and persistent moments m, v; hyper as {@link LayerwiseOptimKernels#lambMoments}. */
    public static void lamb(Tensor p, Tensor g, Tensor m, Tensor v, Tensor hyper, Tensor stats, float lr) {
        try (Grad.Scope ignored = Grad.disabled()) {
            Tensor u = Tensor.result(p.shape(), DType.F32);
            Tensor.record(LAMB_MOMENTS, new Tensor[] { p, g, m, v, hyper, stats(stats) }, new Tensor[] { m, v, u }, null, null, true);
            Tensor norms = Tensor.cat(0, norm(p), norm(u));
            Tensor.record(TRUST_UPDATE, new Tensor[] { p, u, norms }, new Tensor[] { p }, null, new float[] { lr }, true);
        }
    }

    /** One LARS step of p with gradient g and persistent momentum buffer; hyper as {@link LayerwiseOptimKernels#larsUpdate}. */
    public static void lars(Tensor p, Tensor g, Tensor buf, Tensor hyper, Tensor stats) {
        try (Grad.Scope ignored = Grad.disabled()) {
            Tensor norms = Tensor.cat(0, norm(p), norm(g));
            Tensor.record(LARS_UPDATE, new Tensor[] { p, g, buf, norms, hyper, stats(stats) }, new Tensor[] { p, buf }, null, null, true);
        }
    }

    private static FloatArray host(Node n, int i) {
        return (FloatArray) n.in(i).storage().host();
    }

    static final class LambMoments implements Op {

        @Override
        public String name() {
            return "lamb_moments";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray p = e.f32(n.in(0)), g = e.f32(n.in(1)), m = e.f32(n.in(2)), v = e.f32(n.in(3)), h = e.f32(n.in(4)), st = e.f32(n.in(5));
            FloatArray u = e.f32(n.out(2));
            int count = n.in(0).numel();
            e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, LayerwiseOptimKernels::lambMoments, e.context(), p, g, m, v, u, h, st, count));
        }

        @Override
        public void reference(Node n) {
            FloatArray p = host(n, 0), g = host(n, 1), m = host(n, 2), v = host(n, 3), h = host(n, 4), st = host(n, 5);
            FloatArray u = (FloatArray) n.out(2).storage().host();
            for (int i = 0; i < n.in(0).numel(); i++) {
                if (st.get(2) != 0) {
                    u.set(i, 0);
                    continue;
                }
                double grad = g.get(i) * (double) st.get(1);
                double mi = h.get(1) * m.get(i) + (1 - h.get(1)) * grad;
                double vi = h.get(2) * v.get(i) + (1 - h.get(2)) * grad * grad;
                m.set(i, (float) mi);
                v.set(i, (float) vi);
                u.set(i, (float) ((mi / h.get(4)) / (Math.sqrt(vi / h.get(5)) + h.get(3)) + h.get(6) * p.get(i)));
            }
        }
    }

    static final class TrustUpdate implements Op {

        @Override
        public String name() {
            return "trust_update";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray p = e.f32(n.in(0)), u = e.f32(n.in(1)), norms = e.f32(n.in(2));
            float lr = n.fattr[0];
            int count = n.in(0).numel();
            e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, LayerwiseOptimKernels::trustUpdate, e.context(), p, u, norms, lr, count));
        }

        @Override
        public void reference(Node n) {
            FloatArray p = host(n, 0), u = host(n, 1), norms = host(n, 2);
            double r = norms.get(0) > 0 && norms.get(1) > 0 ? norms.get(0) / (double) norms.get(1) : 1;
            for (int i = 0; i < n.in(0).numel(); i++) {
                p.set(i, (float) (p.get(i) - n.fattr[0] * r * u.get(i)));
            }
        }
    }

    static final class LarsUpdate implements Op {

        @Override
        public String name() {
            return "lars_update";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray p = e.f32(n.in(0)), g = e.f32(n.in(1)), buf = e.f32(n.in(2)), norms = e.f32(n.in(3)), h = e.f32(n.in(4)), st = e.f32(n.in(5));
            int count = n.in(0).numel();
            e.task(Emitter.grid1D(count), (tg, id) -> tg.task(id, LayerwiseOptimKernels::larsUpdate, e.context(), p, g, buf, norms, h, st, count));
        }

        @Override
        public void reference(Node n) {
            FloatArray p = host(n, 0), g = host(n, 1), buf = host(n, 2), norms = host(n, 3), h = host(n, 4), st = host(n, 5);
            if (st.get(2) != 0) {
                return;
            }
            double factor = st.get(1);
            double wd = h.get(2);
            double wn = norms.get(0);
            double gn = norms.get(1) * factor;
            double local = wn > 0 && gn > 0 ? h.get(3) * wn / (gn + wd * wn + h.get(4)) : 1;
            for (int i = 0; i < n.in(0).numel(); i++) {
                double w = p.get(i);
                double b = h.get(1) * buf.get(i) + h.get(0) * local * (g.get(i) * factor + wd * w);
                buf.set(i, (float) b);
                p.set(i, (float) (w - b));
            }
        }
    }
}
