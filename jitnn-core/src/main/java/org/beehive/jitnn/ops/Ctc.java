package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.CtcKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * CTC negative log-likelihood per sample (see {@link CtcKernels}). Inputs [logProbs [T, B, C],
 * targets [B, L], inputLengths [B], targetLengths [B]] (INT32 but the first); outputs [loss [B],
 * alpha [B, T, 2L+1]]; iattr = [blank].
 */
public final class Ctc implements Op {

    static final Ctc OP = new Ctc();
    static final Backward BACKWARD = new Backward();

    private Ctc() {
    }

    public static Tensor apply(Tensor logProbs, Tensor targets, Tensor inputLengths, Tensor targetLengths, int blank) {
        Checks.f32(logProbs);
        Checks.rank(logProbs, 3);
        int t = logProbs.dim(0);
        int b = logProbs.dim(1);
        int lmax = targets.dim(1);
        if (2 * lmax + 1 > 1024) {
            throw new IllegalArgumentException("CTC: targets longer than 511 labels are not supported");
        }
        Tensor loss = Tensor.result(Shape.of(b), DType.F32);
        Tensor alpha = Tensor.result(Shape.of(b, t, 2 * lmax + 1), DType.F32);
        Tensor.record(OP, new Tensor[] { logProbs, targets, inputLengths, targetLengths }, new Tensor[] { loss, alpha }, new int[] { blank }, null, true);
        return loss;
    }

    static int group(int lmax) {
        return Math.max(32, Emitter.ceilDiv(2 * lmax + 1, 32) * 32);
    }

    @Override
    public String name() {
        return "ctc";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray lp = e.f32(n.in(0));
        IntArray targets = e.i32(n.in(1));
        IntArray inLen = e.i32(n.in(2));
        IntArray tgtLen = e.i32(n.in(3));
        FloatArray loss = e.f32(n.out(0));
        FloatArray alpha = e.f32(n.out(1));
        int tmax = n.in(0).dim(0), batch = n.in(0).dim(1), classes = n.in(0).dim(2), lmax = n.in(1).dim(1), blank = n.iattr[0];
        e.task(Emitter.blocks1D(batch, group(lmax)), (g, id) -> g.task(id, CtcKernels::forward, e.context(), lp, targets, inLen, tgtLen, alpha, loss, tmax, batch, classes, lmax, blank));
    }

    /** Log-space forward recursion in double precision, written independently of the kernel. */
    @Override
    public void reference(Node n) {
        FloatArray lp = (FloatArray) n.in(0).storage().host();
        IntArray targets = (IntArray) n.in(1).storage().host();
        IntArray inLen = (IntArray) n.in(2).storage().host();
        IntArray tgtLen = (IntArray) n.in(3).storage().host();
        FloatArray loss = (FloatArray) n.out(0).storage().host();
        FloatArray alphaOut = (FloatArray) n.out(1).storage().host();
        int tmax = n.in(0).dim(0), batch = n.in(0).dim(1), classes = n.in(0).dim(2), lmax = n.in(1).dim(1), blank = n.iattr[0];
        int smax = 2 * lmax + 1;
        for (int b = 0; b < batch; b++) {
            int tb = inLen.get(b), sb = 2 * tgtLen.get(b) + 1;
            int[] lab = new int[sb];
            for (int s = 0; s < sb; s++) {
                lab[s] = s % 2 == 0 ? blank : targets.get(b * lmax + s / 2);
            }
            double[][] a = new double[tmax][smax];
            for (double[] row : a) {
                java.util.Arrays.fill(row, Double.NEGATIVE_INFINITY);
            }
            a[0][0] = lp.get(b * classes + lab[0]);
            if (sb > 1) {
                a[0][1] = lp.get(b * classes + lab[1]);
            }
            for (int t = 1; t < tb; t++) {
                for (int s = 0; s < sb; s++) {
                    double v = a[t - 1][s];
                    if (s >= 1) {
                        v = logAdd(v, a[t - 1][s - 1]);
                    }
                    if (s >= 2 && lab[s] != blank && lab[s] != lab[s - 2]) {
                        v = logAdd(v, a[t - 1][s - 2]);
                    }
                    a[t][s] = v + lp.get((t * batch + b) * classes + lab[s]);
                }
            }
            double total = a[tb - 1][sb - 1];
            if (sb >= 2) {
                total = logAdd(total, a[tb - 1][sb - 2]);
            }
            loss.set(b, (float) -total);
            for (int t = 0; t < tmax; t++) {
                for (int s = 0; s < smax; s++) {
                    alphaOut.set((b * tmax + t) * smax + s, (float) Math.max(a[t][s], -1e30));
                }
            }
        }
    }

    static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) {
            return b;
        }
        if (b == Double.NEGATIVE_INFINITY) {
            return a;
        }
        double hi = Math.max(a, b);
        return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        Tensor grad = Tensor.result(n.in(0).shape(), DType.F32);
        Tensor beta = Tensor.result(n.out(1).shape(), DType.F32);
        Tensor.record(BACKWARD, new Tensor[] { n.in(0), n.in(1), n.in(2), n.in(3), n.out(1), n.out(0), gy }, new Tensor[] { grad, beta }, n.iattr.clone(), null, true);
        return new Tensor[] { grad, null, null, null };
    }

    /** inputs [logProbs, targets, inLen, tgtLen, alpha, loss, dLoss]; outputs [dLogProbs, beta]. */
    static final class Backward implements Op {

        @Override
        public String name() {
            return "ctc_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            FloatArray lp = e.f32(n.in(0));
            IntArray targets = e.i32(n.in(1));
            IntArray inLen = e.i32(n.in(2));
            IntArray tgtLen = e.i32(n.in(3));
            FloatArray alpha = e.f32(n.in(4));
            FloatArray loss = e.f32(n.in(5));
            FloatArray gl = e.f32(n.in(6));
            FloatArray grad = e.f32(n.out(0));
            FloatArray beta = e.f32(n.out(1));
            int tmax = n.in(0).dim(0), batch = n.in(0).dim(1), classes = n.in(0).dim(2), lmax = n.in(1).dim(1), blank = n.iattr[0];
            int group = group(lmax);
            e.task(Emitter.blocks1D(batch, group), (g, id) -> g.task(id, CtcKernels::backward, e.context(), lp, targets, inLen, tgtLen, alpha, beta, loss, gl, grad, tmax, batch, classes, lmax,
                    blank, group));
        }

        @Override
        public void reference(Node n) {
            FloatArray lp = (FloatArray) n.in(0).storage().host();
            IntArray targets = (IntArray) n.in(1).storage().host();
            IntArray inLen = (IntArray) n.in(2).storage().host();
            IntArray tgtLen = (IntArray) n.in(3).storage().host();
            FloatArray alpha = (FloatArray) n.in(4).storage().host();
            FloatArray loss = (FloatArray) n.in(5).storage().host();
            FloatArray gl = (FloatArray) n.in(6).storage().host();
            FloatArray grad = (FloatArray) n.out(0).storage().host();
            int tmax = n.in(0).dim(0), batch = n.in(0).dim(1), classes = n.in(0).dim(2), lmax = n.in(1).dim(1), blank = n.iattr[0];
            int smax = 2 * lmax + 1;
            for (int b = 0; b < batch; b++) {
                int tb = inLen.get(b), sb = 2 * tgtLen.get(b) + 1;
                int[] lab = new int[sb];
                for (int s = 0; s < sb; s++) {
                    lab[s] = s % 2 == 0 ? blank : targets.get(b * lmax + s / 2);
                }
                double[][] be = new double[tmax][smax];
                for (double[] row : be) {
                    java.util.Arrays.fill(row, Double.NEGATIVE_INFINITY);
                }
                for (int t = tb - 1; t >= 0; t--) {
                    for (int s = 0; s < sb; s++) {
                        double l = lp.get((t * batch + b) * classes + lab[s]);
                        if (t == tb - 1) {
                            be[t][s] = s >= sb - 2 ? l : Double.NEGATIVE_INFINITY;
                            continue;
                        }
                        double v = be[t + 1][s];
                        if (s + 1 < sb) {
                            v = logAdd(v, be[t + 1][s + 1]);
                        }
                        if (s + 2 < sb && lab[s] != blank && lab[s] != lab[s + 2]) {
                            v = logAdd(v, be[t + 1][s + 2]);
                        }
                        be[t][s] = v + l;
                    }
                }
                double logp = -loss.get(b);
                for (int t = 0; t < tmax; t++) {
                    for (int c = 0; c < classes; c++) {
                        int at = (t * batch + b) * classes + c;
                        if (t >= tb) {
                            grad.set(at, 0);
                            continue;
                        }
                        double acc = Double.NEGATIVE_INFINITY;
                        for (int s = 0; s < sb; s++) {
                            if (lab[s] == c) {
                                double av = alpha.get((b * tmax + t) * smax + s);
                                acc = logAdd(acc, (av <= -1e29 ? Double.NEGATIVE_INFINITY : av) + be[t][s] - lp.get(at));
                            }
                        }
                        grad.set(at, (float) (gl.get(b) * (Math.exp(lp.get(at)) - Math.exp(acc - logp))));
                    }
                }
            }
        }
    }
}
