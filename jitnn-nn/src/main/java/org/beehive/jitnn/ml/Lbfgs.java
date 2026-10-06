package org.beehive.jitnn.ml;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.function.Function;

import org.beehive.jitnn.Tensor;

/**
 * Limited-memory BFGS (Nocedal and Wright, algorithm 7.4) with a backtracking Armijo line search,
 * for the smooth convex objectives of the linear models. The parameters are a small host vector;
 * each evaluation runs the objective and its gradient on the device through autograd.
 */
final class Lbfgs {

    private Lbfgs() {
    }

    /** The objective of a parameter tensor (requiring its gradient), as a scalar tensor. */
    static float[] minimise(Function<Tensor, Tensor> objective, float[] start, int[] shape, int maxIter, double tol, int memory) {
        int n = start.length;
        double[] x = toDouble(start);
        double[][] fg = evaluate(objective, x, shape);
        Deque<double[][]> pairs = new ArrayDeque<>();
        for (int it = 0; it < maxIter; it++) {
            double[] g = fg[1];
            if (maxAbs(g) <= tol) {
                break;
            }
            // two-loop recursion: d = -H g
            double[] q = g.clone();
            double[] alphas = new double[pairs.size()];
            int i = 0;
            for (var it2 = pairs.descendingIterator(); it2.hasNext(); i++) {
                double[][] p = it2.next();
                alphas[i] = dot(p[0], q) / dot(p[1], p[0]);
                axpy(-alphas[i], p[1], q);
            }
            if (!pairs.isEmpty()) {
                double[][] last = pairs.peekLast();
                double gamma = dot(last[0], last[1]) / dot(last[1], last[1]);
                for (int k = 0; k < n; k++) {
                    q[k] *= gamma;
                }
            }
            i = pairs.size() - 1;
            for (double[][] p : pairs) {
                double beta = dot(p[1], q) / dot(p[1], p[0]);
                axpy(alphas[i--] - beta, p[0], q);
            }
            double[] d = new double[n];
            for (int k = 0; k < n; k++) {
                d[k] = -q[k];
            }
            double slope = dot(g, d);
            if (slope >= 0) {
                for (int k = 0; k < n; k++) {
                    d[k] = -g[k];
                }
                slope = -dot(g, g);
                pairs.clear();
            }
            double step = pairs.isEmpty() && it == 0 ? Math.min(1.0, 1.0 / Math.max(maxAbs(g), 1e-12)) : 1.0;
            double[] xn = null;
            double[][] fgn = null;
            for (int ls = 0; ls < 40; ls++) {
                xn = x.clone();
                axpy(step, d, xn);
                fgn = evaluate(objective, xn, shape);
                if (fgn[0][0] <= fg[0][0] + 1e-4 * step * slope) {
                    break;
                }
                step *= 0.5;
            }
            double[] s = new double[n];
            double[] y = new double[n];
            for (int k = 0; k < n; k++) {
                s[k] = xn[k] - x[k];
                y[k] = fgn[1][k] - g[k];
            }
            boolean stalled = Math.abs(fg[0][0] - fgn[0][0]) <= 1e-12 * Math.max(1, Math.abs(fg[0][0]));
            x = xn;
            fg = fgn;
            if (dot(s, y) > 1e-12) {
                pairs.addLast(new double[][] { s, y });
                if (pairs.size() > memory) {
                    pairs.removeFirst();
                }
            }
            if (stalled) {
                break;
            }
        }
        float[] out = new float[n];
        for (int k = 0; k < n; k++) {
            out[k] = (float) x[k];
        }
        return out;
    }

    private static double[][] evaluate(Function<Tensor, Tensor> objective, double[] x, int[] shape) {
        float[] xf = new float[x.length];
        for (int k = 0; k < x.length; k++) {
            xf[k] = (float) x[k];
        }
        Tensor p = Tensor.of(xf, shape).requiresGrad(true);
        Tensor f = objective.apply(p);
        f.backward();
        double value = f.item();
        return new double[][] { { value }, toDouble(p.grad().toFloatArray()) };
    }

    private static double[] toDouble(float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i];
        }
        return d;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static void axpy(double a, double[] x, double[] y) {
        for (int i = 0; i < x.length; i++) {
            y[i] += a * x[i];
        }
    }

    private static double maxAbs(double[] v) {
        double m = 0;
        for (double x : v) {
            m = Math.max(m, Math.abs(x));
        }
        return m;
    }
}
