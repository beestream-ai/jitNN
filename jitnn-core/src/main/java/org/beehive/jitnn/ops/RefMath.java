package org.beehive.jitnn.ops;

import org.beehive.jitnn.kernels.ElementwiseKernels;

/**
 * Double-precision versions of the element-wise ops, for the host reference. Written
 * independently of the kernels so they can serve as an oracle for them.
 */
final class RefMath {

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
            default -> alpha;
        };
    }
}
