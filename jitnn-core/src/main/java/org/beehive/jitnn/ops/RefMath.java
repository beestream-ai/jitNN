package org.beehive.jitnn.ops;

import org.beehive.jitnn.kernels.ElementwiseKernels;

/**
 * Double-precision versions of the element-wise ops, for the host reference. Written
 * independently of the kernels so they can serve as an oracle for them.
 */
final class RefMath {

    static final double SELU_ALPHA = 1.6732632423543772;
    static final double SELU_SCALE = 1.0507009873554805;

    private RefMath() {
    }

    static double binary(int op, double x, double y, double alpha, double beta) {
        return switch (op) {
            case ElementwiseKernels.ADD -> x + y;
            case ElementwiseKernels.SUB -> x - y;
            case ElementwiseKernels.MUL -> x * y;
            case ElementwiseKernels.DIV -> x / y;
            case ElementwiseKernels.MAX -> Math.max(x, y);
            case ElementwiseKernels.MIN -> Math.min(x, y);
            case ElementwiseKernels.POW -> Math.pow(x, y);
            case ElementwiseKernels.RELU_GRAD -> x > 0 ? y : 0;
            case ElementwiseKernels.SIGN_MUL -> Math.signum(x) * y;
            case ElementwiseKernels.CLAMP_GRAD -> x >= alpha && x <= beta ? y : 0;
            case ElementwiseKernels.GT -> x > y ? 1 : 0;
            case ElementwiseKernels.LT -> x < y ? 1 : 0;
            case ElementwiseKernels.GE -> x >= y ? 1 : 0;
            case ElementwiseKernels.EQ -> x == y ? 1 : 0;
            case ElementwiseKernels.SIGMOID_GRAD -> y * x * (1 - x);
            case ElementwiseKernels.TANH_GRAD -> y * (1 - x * x);
            case ElementwiseKernels.GELU_GRAD -> y * (normalCdf(x) + x * Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI));
            case ElementwiseKernels.GELU_TANH_GRAD -> {
                double c = Math.sqrt(2 / Math.PI);
                double t = Math.tanh(c * (x + 0.044715 * x * x * x));
                yield y * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x));
            }
            case ElementwiseKernels.SILU_GRAD -> {
                double s = 1 / (1 + Math.exp(-x));
                yield y * s * (1 + x * (1 - s));
            }
            case ElementwiseKernels.LEAKY_RELU_GRAD -> x > 0 ? y : alpha * y;
            case ElementwiseKernels.SOFTPLUS_GRAD -> y / (1 + Math.exp(-x));
            case ElementwiseKernels.BCE -> -(y * Math.max(Math.log(x), -100) + (1 - y) * Math.max(Math.log(1 - x), -100));
            case ElementwiseKernels.BCE_DX -> (x - y) / Math.max(x * (1 - x), 1e-12);
            case ElementwiseKernels.BCE_DY -> Math.max(Math.log(1 - x), -100) - Math.max(Math.log(x), -100);
            case ElementwiseKernels.BCE_LOGITS -> (1 - y) * x + Math.max(-x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
            case ElementwiseKernels.BCE_LOGITS_DX -> 1 / (1 + Math.exp(-x)) - y;
            case ElementwiseKernels.ADD_RELU -> Math.max(x + y, 0);
            case ElementwiseKernels.ELU_GRAD -> x > 0 ? y : y * alpha * Math.exp(x);
            case ElementwiseKernels.SELU_GRAD -> x > 0 ? y * SELU_SCALE : y * SELU_SCALE * SELU_ALPHA * Math.exp(x);
            case ElementwiseKernels.CELU_GRAD -> x > 0 ? y : y * Math.exp(x / alpha);
            case ElementwiseKernels.MISH_GRAD -> {
                double t = Math.tanh(Math.log1p(Math.exp(x)));
                yield y * (t + x / (1 + Math.exp(-x)) * (1 - t * t));
            }
            case ElementwiseKernels.SOFTSIGN_GRAD -> y / ((1 + Math.abs(x)) * (1 + Math.abs(x)));
            case ElementwiseKernels.HARD_SIGMOID_GRAD -> x > -3 && x < 3 ? y / 6 : 0;
            case ElementwiseKernels.HARD_SWISH_GRAD -> x < -3 ? 0 : (x > 3 ? y : y * (2 * x + 3) / 6);
            case ElementwiseKernels.THRESHOLD_GRAD -> x > alpha ? y : 0;
            case ElementwiseKernels.PRELU -> x > 0 ? x : y * x;
            case ElementwiseKernels.PRELU_SLOPE -> x > 0 ? 1 : y;
            case ElementwiseKernels.ATAN2 -> Math.atan2(x, y);
            case ElementwiseKernels.NE -> x != y ? 1 : 0;
            case ElementwiseKernels.LE -> x <= y ? 1 : 0;
            case ElementwiseKernels.LOGICAL_AND -> x != 0 && y != 0 ? 1 : 0;
            case ElementwiseKernels.LOGICAL_OR -> x != 0 || y != 0 ? 1 : 0;
            case ElementwiseKernels.LOGICAL_XOR -> (x != 0) != (y != 0) ? 1 : 0;
            default -> x;
        };
    }

    static double unary(int op, double x, double alpha, double beta) {
        return switch (op) {
            case ElementwiseKernels.NEG -> -x;
            case ElementwiseKernels.EXP -> Math.exp(x);
            case ElementwiseKernels.LOG -> Math.log(x);
            case ElementwiseKernels.SQRT -> Math.sqrt(x);
            case ElementwiseKernels.ABS -> Math.abs(x);
            case ElementwiseKernels.RELU -> x > 0 ? x : 0;
            case ElementwiseKernels.AFFINE -> alpha * x + beta;
            case ElementwiseKernels.POW_SCALAR -> Math.pow(x, alpha);
            case ElementwiseKernels.CLAMP -> Math.min(Math.max(x, alpha), beta);
            case ElementwiseKernels.RECIPROCAL -> 1.0 / x;
            case ElementwiseKernels.SQUARE -> x * x;
            case ElementwiseKernels.TANH -> Math.tanh(x);
            case ElementwiseKernels.SIGMOID -> 1 / (1 + Math.exp(-x));
            case ElementwiseKernels.GELU -> x * normalCdf(x);
            case ElementwiseKernels.GELU_TANH -> 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
            case ElementwiseKernels.SILU -> x / (1 + Math.exp(-x));
            case ElementwiseKernels.LEAKY_RELU -> x > 0 ? x : alpha * x;
            case ElementwiseKernels.SOFTPLUS -> Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
            case ElementwiseKernels.ELU -> x > 0 ? x : alpha * Math.expm1(x);
            case ElementwiseKernels.SELU -> x > 0 ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * Math.expm1(x);
            case ElementwiseKernels.CELU -> x > 0 ? x : alpha * Math.expm1(x / alpha);
            case ElementwiseKernels.MISH -> x * Math.tanh(Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))));
            case ElementwiseKernels.SOFTSIGN -> x / (1 + Math.abs(x));
            case ElementwiseKernels.HARD_SIGMOID -> Math.min(Math.max(x / 6 + 0.5, 0), 1);
            case ElementwiseKernels.HARD_SWISH -> x * Math.min(Math.max(x / 6 + 0.5, 0), 1);
            case ElementwiseKernels.THRESHOLD -> x > alpha ? x : 0;
            case ElementwiseKernels.SIN -> Math.sin(x);
            case ElementwiseKernels.COS -> Math.cos(x);
            case ElementwiseKernels.TAN -> Math.tan(x);
            case ElementwiseKernels.ASIN -> Math.asin(x);
            case ElementwiseKernels.ACOS -> Math.acos(x);
            case ElementwiseKernels.ATAN -> Math.atan(x);
            case ElementwiseKernels.SINH -> Math.sinh(x);
            case ElementwiseKernels.COSH -> Math.cosh(x);
            case ElementwiseKernels.ASINH -> Math.log(x + Math.sqrt(x * x + 1));
            case ElementwiseKernels.ACOSH -> Math.log(x + Math.sqrt(x * x - 1));
            case ElementwiseKernels.ATANH -> 0.5 * Math.log((1 + x) / (1 - x));
            case ElementwiseKernels.ERF -> 1 - erfc(x);
            case ElementwiseKernels.ERFC -> erfc(x);
            case ElementwiseKernels.ROUND -> Math.rint(x);
            case ElementwiseKernels.TRUNC -> x < 0 ? Math.ceil(x) : Math.floor(x);
            case ElementwiseKernels.FLOOR -> Math.floor(x);
            case ElementwiseKernels.CEIL -> Math.ceil(x);
            case ElementwiseKernels.SIGN -> Math.signum(x);
            case ElementwiseKernels.LOG1P -> Math.log1p(x);
            case ElementwiseKernels.EXPM1 -> Math.expm1(x);
            case ElementwiseKernels.LOGICAL_NOT -> x == 0 ? 1 : 0;
            default -> alpha;
        };
    }

    static double normalCdf(double x) {
        return 0.5 * erfc(-x / Math.sqrt(2));
    }

    /** erfc in double: Taylor series of erf for |x| < 3, Lentz-free continued fraction beyond. */
    static double erfc(double x) {
        double z = Math.abs(x);
        double r;
        if (z < 3) {
            double term = z;
            double sum = z;
            for (int n = 1; n < 200 && Math.abs(term) > 1e-17 * Math.abs(sum); n++) {
                term *= -z * z / n;
                sum += term / (2 * n + 1);
            }
            r = 1 - 2 / Math.sqrt(Math.PI) * sum;
        } else {
            double f = z;
            for (int k = 120; k >= 1; k--) {
                f = z + (k / 2.0) / f;
            }
            r = Math.exp(-z * z) / (Math.sqrt(Math.PI) * f);
        }
        return x >= 0 ? r : 2 - r;
    }
}
