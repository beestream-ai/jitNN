package org.beehive.jitnn.ml;

import org.beehive.jitnn.Linalg;
import org.beehive.jitnn.Tensor;

/**
 * Least squares with an optional L2 penalty alpha (ridge regression), solved in closed form from
 * the normal equations (X^T X + alpha I) w = X^T y on centred data, after scikit-learn's
 * LinearRegression (alpha 0) and Ridge.
 */
public final class LinearRegression {

    private final float alpha;
    private final boolean fitIntercept;
    private Tensor coef;
    private Tensor intercept;

    public LinearRegression(float alpha, boolean fitIntercept) {
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
    }

    public LinearRegression() {
        this(0.0f, true);
    }

    /** y [n] or [n, t]. */
    public LinearRegression fit(Tensor x, Tensor y) {
        Ml.check(x, "LinearRegression");
        int d = x.dim(1);
        Tensor y2 = y.rank() == 1 ? y.unsqueeze(1) : y;
        Tensor xm = fitIntercept ? x.mean(0, true) : null;
        Tensor ym = fitIntercept ? y2.mean(0, true) : null;
        Tensor xc = fitIntercept ? x.sub(xm) : x;
        Tensor yc = fitIntercept ? y2.sub(ym) : y2;
        float[] eye = new float[d * d];
        for (int i = 0; i < d; i++) {
            eye[i * d + i] = alpha;
        }
        Tensor a = xc.matmul(xc, true, false).add(Tensor.of(eye, d, d));
        Tensor w = Linalg.solve(a, xc.matmul(yc, true, false));
        coef = w.transpose(0, 1).mul(1.0f);
        intercept = fitIntercept ? ym.sub(xm.matmul(w)).reshape(y2.dim(1)) : Tensor.zeros(y2.dim(1));
        if (y.rank() == 1) {
            coef = coef.reshape(d);
            intercept = intercept.reshape(1);
        }
        return this;
    }

    /** Coefficients [d] (one target) or [t, d]. */
    public Tensor coef() {
        return coef;
    }

    public Tensor intercept() {
        return intercept;
    }

    public Tensor predict(Tensor x) {
        if (coef.rank() == 1) {
            return x.matmul(coef.unsqueeze(1)).reshape(x.dim(0)).add(intercept);
        }
        return x.matmul(coef, false, true).add(intercept);
    }
}
