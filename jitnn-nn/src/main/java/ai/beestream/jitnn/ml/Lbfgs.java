/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.ml;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.function.Function;

import ai.beestream.jitnn.Tensor;

/**
 * Limited-memory BFGS (Nocedal and Wright, algorithm 7.4) for the smooth convex objectives of the
 * linear models, with a line search on the directional derivative (FP32 objective values are too
 * coarse to compare near the optimum). Stops when the largest gradient entry is at most tol. The parameters are a small host vector;
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
            // line search on the directional derivative phi'(a) = g(x + a d) . d, which FP32 resolves
            // long after differences of the objective stop resolving: strong Wolfe curvature
            double lo = 0;
            double hi = Double.POSITIVE_INFINITY;
            double step = pairs.isEmpty() ? Math.min(1.0, 1.0 / Math.max(maxAbs(g), 1e-12)) : 1.0;
            double[] xn = null;
            double[][] fgn = null;
            for (int ls = 0; ls < 40; ls++) {
                xn = x.clone();
                axpy(step, d, xn);
                fgn = evaluate(objective, xn, shape);
                double deriv = dot(fgn[1], d);
                boolean worse = fgn[0][0] > fg[0][0] + 1e-5 * Math.abs(fg[0][0]);
                if (!worse && Math.abs(deriv) <= 0.9 * Math.abs(slope)) {
                    break;
                }
                if (worse || deriv > 0) {
                    hi = step;
                } else {
                    lo = step;
                }
                step = Double.isInfinite(hi) ? 2 * step : 0.5 * (lo + hi);
            }
            double[] s = new double[n];
            double[] y = new double[n];
            for (int k = 0; k < n; k++) {
                s[k] = xn[k] - x[k];
                y[k] = fgn[1][k] - g[k];
            }
            x = xn;
            fg = fgn;
            if (dot(s, y) > 1e-20) {
                pairs.addLast(new double[][] { s, y });
                if (pairs.size() > memory) {
                    pairs.removeFirst();
                }
            }
            if (Math.sqrt(dot(s, s)) <= 1e-12 * Math.max(1.0, Math.sqrt(dot(x, x)))) {
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
