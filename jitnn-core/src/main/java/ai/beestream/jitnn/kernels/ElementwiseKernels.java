package ai.beestream.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Element-wise kernels. One kernel per arity, selected by an op code: the branch is uniform
 * across a launch, so its cost is negligible next to the memory traffic, and one compiled kernel
 * serves every op of that arity.
 *
 * <p>Kernel code uses non-short-circuit {@code &} and {@code |} in conditions (TornadoVM bug T1:
 * short-circuit operators inside loops can drop a branch).
 */
public final class ElementwiseKernels {

    // Binary op codes: out = f(a, b).
    public static final int ADD = 0;
    public static final int SUB = 1;
    public static final int MUL = 2;
    public static final int DIV = 3;
    public static final int MAX = 4;
    public static final int MIN = 5;
    public static final int POW = 6;
    /** a = x, b = dy: dy where x > 0. */
    public static final int RELU_GRAD = 7;
    /** sign(a) * b. */
    public static final int SIGN_MUL = 8;
    /** a = x, b = dy: dy where alpha &lt;= x &lt;= beta. */
    public static final int CLAMP_GRAD = 9;
    public static final int GT = 10;
    public static final int LT = 11;
    public static final int GE = 12;
    public static final int EQ = 13;
    /** a (b is ignored): broadcast copy. */
    public static final int FIRST = 14;
    /** a = y = sigmoid(x), b = dy. */
    public static final int SIGMOID_GRAD = 15;
    /** a = y = tanh(x), b = dy. */
    public static final int TANH_GRAD = 16;
    /** a = x, b = dy, for GELU with the exact (erf) form. */
    public static final int GELU_GRAD = 17;
    /** a = x, b = dy, for GELU with the tanh approximation. */
    public static final int GELU_TANH_GRAD = 18;
    /** a = x, b = dy, for SiLU (x * sigmoid(x)). */
    public static final int SILU_GRAD = 19;
    /** y where x > 0, else alpha * y (x is the input). */
    public static final int LEAKY_RELU_GRAD = 20;
    /** y * sigmoid(x) (x is the input). */
    public static final int SOFTPLUS_GRAD = 21;
    /** Binary cross-entropy of probability x against target y, logs clamped at -100 (PyTorch). */
    public static final int BCE = 22;
    /** d BCE / d x: (x - y) / max(x (1 - x), 1e-12). */
    public static final int BCE_DX = 23;
    /** d BCE / d y: log(1 - x) - log(x), both clamped at -100. */
    public static final int BCE_DY = 24;
    /** Binary cross-entropy of logit x against target y: (1 - y) x + max(-x, 0) + log1p(exp(-|x|)). */
    public static final int BCE_LOGITS = 25;
    /** d BCE_LOGITS / d x: sigmoid(x) - y. */
    public static final int BCE_LOGITS_DX = 26;
    /** relu(x + y): a residual connection and its activation in one pass. */
    public static final int ADD_RELU = 27;
    /** y where x > 0, else y alpha exp(x) (x is the input; alpha in alpha). */
    public static final int ELU_GRAD = 28;
    /** SELU's derivative times y. */
    public static final int SELU_GRAD = 29;
    /** y where x > 0, else y exp(x / alpha). */
    public static final int CELU_GRAD = 30;
    /** Mish's derivative times y. */
    public static final int MISH_GRAD = 31;
    /** y / (1 + |x|)^2. */
    public static final int SOFTSIGN_GRAD = 32;
    /** y / 6 inside (-3, 3), else 0. */
    public static final int HARD_SIGMOID_GRAD = 33;
    /** y times HardSwish's derivative. */
    public static final int HARD_SWISH_GRAD = 34;
    /** y where x > alpha, else 0. */
    public static final int THRESHOLD_GRAD = 35;
    /** x where x > 0, else y x (y is the slope). */
    public static final int PRELU = 36;
    /** 1 where x > 0, else y (the slope): dPReLU/dx. */
    public static final int PRELU_SLOPE = 37;
    // comparisons and logic give 0 or 1
    public static final int ATAN2 = 38;
    public static final int NE = 39;
    public static final int LE = 40;
    public static final int LOGICAL_AND = 41;
    public static final int LOGICAL_OR = 42;
    public static final int LOGICAL_XOR = 43;
    /** max(x, y), NaN when either is NaN (torch.maximum). */
    public static final int MAXIMUM = 44;
    /** min(x, y), NaN when either is NaN (torch.minimum). */
    public static final int MINIMUM = 45;
    /** max(x, y) ignoring a NaN operand (torch.fmax). */
    public static final int FMAX = 46;
    /** min(x, y) ignoring a NaN operand (torch.fmin). */
    public static final int FMIN = 47;
    /** x - floor(x / y) y, the sign of y (torch.remainder, Python's %). */
    public static final int REMAINDER = 48;
    /** x - trunc(x / y) y, the sign of x (torch.fmod, C's fmod). */
    public static final int FMOD = 49;

    // Unary op codes: out = f(x; alpha, beta).
    public static final int NEG = 0;
    public static final int EXP = 1;
    public static final int LOG = 2;
    public static final int SQRT = 3;
    public static final int ABS = 4;
    public static final int RELU = 5;
    /** alpha * x + beta. */
    public static final int AFFINE = 6;
    /** x ^ alpha. */
    public static final int POW_SCALAR = 7;
    /** clamp(x, alpha, beta). */
    public static final int CLAMP = 8;
    public static final int RECIPROCAL = 9;
    public static final int SQUARE = 10;
    /** Fills with alpha. */
    public static final int FILL = 11;
    public static final int TANH = 12;
    public static final int SIGMOID = 13;
    /** x * Phi(x), Phi the standard normal CDF (PyTorch's approximate='none'). */
    public static final int GELU = 14;
    /** 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) (PyTorch's approximate='tanh'). */
    public static final int GELU_TANH = 15;
    /** x * sigmoid(x), also called swish. */
    public static final int SILU = 16;
    /** x where x > 0, else alpha * x. */
    public static final int LEAKY_RELU = 17;
    /** log(1 + exp(x)), overflow-free. */
    public static final int SOFTPLUS = 18;
    /** alpha (expm1(x)) for x &lt;= 0. */
    public static final int ELU = 19;
    /** SELU's scale (x, or alpha expm1(x) for x &lt;= 0), PyTorch's constants. */
    public static final int SELU = 20;
    /** max(0, x) + min(0, alpha expm1(x / alpha)). */
    public static final int CELU = 21;
    /** x tanh(softplus(x)). */
    public static final int MISH = 22;
    /** x / (1 + |x|). */
    public static final int SOFTSIGN = 23;
    /** clamp(x / 6 + 1/2, 0, 1). */
    public static final int HARD_SIGMOID = 24;
    /** x clamp(x / 6 + 1/2, 0, 1). */
    public static final int HARD_SWISH = 25;
    /** x where x > alpha, else 0 (Keras's ThresholdedReLU). */
    public static final int THRESHOLD = 26;
    // math and logic; gradients are composed in Unary.backward
    public static final int SIN = 27;
    public static final int COS = 28;
    public static final int TAN = 29;
    public static final int ASIN = 30;
    public static final int ACOS = 31;
    public static final int ATAN = 32;
    public static final int SINH = 33;
    public static final int COSH = 34;
    public static final int ASINH = 35;
    public static final int ACOSH = 36;
    public static final int ATANH = 37;
    public static final int ERF = 38;
    public static final int ERFC = 39;
    public static final int ROUND = 40;
    public static final int TRUNC = 41;
    public static final int FLOOR = 42;
    public static final int CEIL = 43;
    public static final int SIGN = 44;
    public static final int LOG1P = 45;
    public static final int EXPM1 = 46;
    public static final int LOGICAL_NOT = 47;
    // NaN-aware: tested by bit pattern, as TornadoVM folds x != x
    public static final int IS_NAN = 48;
    public static final int IS_INF = 49;
    public static final int IS_FINITE = 50;
    /** NaN to alpha, +inf to beta, -inf to -beta. */
    public static final int NAN_TO_NUM = 51;
    /**
     * Quantise and dequantise in one step, as ONNX's QuantizeLinear then DequantizeLinear:
     * clamp(round_half_even(x / alpha), beta, beta + 255) * alpha, beta being the lowest code minus
     * the zero point. FAKE_QUANT16: a range of 65535.
     */
    public static final int FAKE_QUANT8 = 52;
    public static final int FAKE_QUANT16 = 53;

    private static final float SQRT_HALF = 0.70710678f;
    private static final float INV_SQRT_2PI = 0.39894228f;
    private static final float SQRT_2_OVER_PI = 0.79788456f;
    private static final float GELU_CUBIC = 0.044715f;
    private static final float SELU_ALPHA = 1.6732632423543772f;
    private static final float SELU_SCALE = 1.0507009873554805f;

    private ElementwiseKernels() {
    }

    /**
     * Complementary error function, Numerical Recipes' erfcc: Chebyshev fit with a fractional
     * error below 1.2e-7 everywhere. WORKAROUND(TornadoVM): TornadoMath has no erf/erfc.
     */
    public static float erfc(float x) {
        float z = TornadoMath.abs(x);
        float t = 1.0f / (1.0f + 0.5f * z);
        float poly = -1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
                + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
        float r = t * TornadoMath.exp(-z * z + poly);
        return x >= 0.0f ? r : 2.0f - r;
    }

    /**
     * Overflow-free tanh, through exp(-2|x|) in (0, 1]. WORKAROUND(TornadoVM): Metal's tanh returns NaN
     * for large arguments (e.g. 66).
     */
    /** GELU with the tanh approximation, 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))). */
    public static float geluTanh(float x) {
        return 0.5f * x * (1.0f + tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)));
    }

    public static float tanh(float x) {
        float e = TornadoMath.exp(-2.0f * TornadoMath.abs(x));
        float t = (1.0f - e) / (1.0f + e);
        return x >= 0.0f ? t : -t;
    }

    /** Overflow-free logistic function. */
    public static float sigmoid(float x) {
        float r;
        if (x >= 0.0f) {
            r = 1.0f / (1.0f + TornadoMath.exp(-x));
        } else {
            float e = TornadoMath.exp(x);
            r = e / (1.0f + e);
        }
        return r;
    }

    /**
     * log(1 + z) for z in [0, 1], accurate for small z. WORKAROUND(TornadoVM): TornadoMath has no log1p;
     * the rounding error of u = 1 + z is cancelled by z / (u - 1) (Goldberg's trick).
     */
    public static float log1p(float z) {
        float u = 1.0f + z;
        float d = u - 1.0f;
        return d == 0.0f ? z : TornadoMath.log(u) * (z / d);
    }

    /**
     * exp(x) - 1, accurate near 0 (Kahan's trick: the rounding of u = exp(x) cancels in
     * (u - 1) x / log(u)). WORKAROUND(TornadoVM): TornadoMath has no expm1.
     */
    public static float expm1(float x) {
        float u = TornadoMath.exp(x);
        float r;
        if (u == 1.0f) {
            r = x;
        } else if (u - 1.0f == -1.0f) {
            r = -1.0f;
        } else {
            r = (u - 1.0f) * x / TornadoMath.log(u);
        }
        return r;
    }

    /** Overflow-free softplus: max(x, 0) + log1p(exp(-|x|)). */
    public static float softplus(float x) {
        return (x > 0.0f ? x : 0.0f) + log1p(TornadoMath.exp(-TornadoMath.abs(x)));
    }

    /** log(x) clamped below at -100, as PyTorch's binary cross-entropy. */
    public static float clampedLog(float x) {
        float r = -100.0f;
        if (x > 0.0f) {
            float l = TornadoMath.log(x);
            r = l > -100.0f ? l : -100.0f;
        }
        return r;
    }

    /** erf: an odd series below 0.5 (where 1 - erfc would cancel), 1 - erfc(x) beyond. */
    public static float erf(float x) {
        float a = TornadoMath.abs(x);
        float r;
        if (a < 0.5f) {
            float x2 = x * x;
            r = 1.1283791671f * x * (1.0f - x2 * (1.0f / 3.0f - x2 * (0.1f - x2 * (1.0f / 42.0f - x2 * (1.0f / 216.0f - x2 * (1.0f / 1320.0f - x2 / 9360.0f))))));
        } else {
            r = 1.0f - erfc(a);
            r = x < 0.0f ? -r : r;
        }
        return r;
    }

    /** NaN by bit pattern: every exponent bit set and a non-zero mantissa. */
    public static boolean isNaN(float x) {
        int bits = Float.floatToRawIntBits(x);
        return ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0);
    }

    /** Infinity by bit pattern: every exponent bit set and a zero mantissa. */
    public static boolean isInf(float x) {
        int bits = Float.floatToRawIntBits(x);
        return ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) == 0);
    }

    /** See {@link #FAKE_QUANT8}: the quantisation steps lo .. lo + range, back in x's units. */
    public static float fakeQuant(float x, float scale, float lo, float range) {
        float q = roundHalfEven(x / scale);
        float hi = lo + range;
        q = q < lo ? lo : (q > hi ? hi : q);
        return q * scale;
    }

    /** The NaN-aware unary ops. */
    public static float unaryNaN(int op, float x, float alpha, float beta) {
        float r;
        if (op == IS_NAN) {
            r = isNaN(x) ? 1.0f : 0.0f;
        } else if (op == IS_INF) {
            r = isInf(x) ? 1.0f : 0.0f;
        } else if (op == IS_FINITE) {
            r = (Float.floatToRawIntBits(x) & 0x7F800000) != 0x7F800000 ? 1.0f : 0.0f;
        } else if (op == NAN_TO_NUM) {
            r = isNaN(x) ? alpha : (isInf(x) ? (x > 0.0f ? beta : -beta) : x);
        } else if (op == FAKE_QUANT8) {
            r = fakeQuant(x, alpha, beta, 255.0f);
        } else if (op == FAKE_QUANT16) {
            r = fakeQuant(x, alpha, beta, 65535.0f);
        } else {
            r = x;
        }
        return r;
    }

    /** The NaN-aware binary ops. */
    public static float binaryNaN(int op, float x, float y, float alpha, float beta) {
        boolean nx = isNaN(x);
        boolean ny = isNaN(y);
        float r;
        if (op == MAXIMUM) {
            r = nx ? x : (ny ? y : (x > y ? x : y));
        } else if (op == MINIMUM) {
            r = nx ? x : (ny ? y : (x < y ? x : y));
        } else if (op == FMAX) {
            r = nx ? y : (ny ? x : (x > y ? x : y));
        } else if (op == FMIN) {
            r = nx ? y : (ny ? x : (x < y ? x : y));
        } else if (op == REMAINDER) {
            r = x - TornadoMath.floor(x / y) * y;
        } else if (op == FMOD) {
            float q = x / y;
            float t = q < 0.0f ? TornadoMath.ceil(q) : TornadoMath.floor(q);
            r = x - t * y;
        } else {
            r = x;
        }
        return r;
    }

    /** Round half to even (PyTorch's round). */
    public static float roundHalfEven(float x) {
        float f = TornadoMath.floor(x);
        float d = x - f;
        float r;
        if (d > 0.5f) {
            r = f + 1.0f;
        } else if (d < 0.5f) {
            r = f;
        } else {
            float half = f * 0.5f;
            r = half == TornadoMath.floor(half) ? f : f + 1.0f;
        }
        return r;
    }

    /** Standard normal CDF, through erfc so it keeps relative precision in the left tail. */
    public static float normalCdf(float x) {
        return 0.5f * erfc(-x * SQRT_HALF);
    }

    /** Binary op {@code op}; kernels branch to the groups themselves, as for {@link #unaryOp}. */
    public static float binaryOp(int op, float x, float y, float alpha, float beta) {
        float r;
        if (op >= MAXIMUM) {
            r = binaryNaN(op, x, y, alpha, beta);
        } else if (op >= ATAN2) {
            r = binaryMath(op, x, y, alpha, beta);
        } else if (op >= ELU_GRAD) {
            r = binaryActivationGrad(op, x, y, alpha, beta);
        } else {
            r = binaryBase(op, x, y, alpha, beta);
        }
        return r;
    }

    /** Binary ops of waves 1-4. */
    public static float binaryBase(int op, float x, float y, float alpha, float beta) {
        float r;
        if (op == ADD) {
            r = x + y;
        } else if (op == SUB) {
            r = x - y;
        } else if (op == MUL) {
            r = x * y;
        } else if (op == DIV) {
            r = x / y;
        } else if (op == MAX) {
            r = x > y ? x : y;
        } else if (op == MIN) {
            r = x < y ? x : y;
        } else if (op == POW) {
            r = TornadoMath.pow(x, y);
        } else if (op == RELU_GRAD) {
            r = x > 0.0f ? y : 0.0f;
        } else if (op == SIGN_MUL) {
            r = x > 0.0f ? y : (x < 0.0f ? -y : 0.0f);
        } else if (op == CLAMP_GRAD) {
            r = (x >= alpha) & (x <= beta) ? y : 0.0f;
        } else if (op == GT) {
            r = x > y ? 1.0f : 0.0f;
        } else if (op == LT) {
            r = x < y ? 1.0f : 0.0f;
        } else if (op == GE) {
            r = x >= y ? 1.0f : 0.0f;
        } else if (op == EQ) {
            r = x == y ? 1.0f : 0.0f;
        } else if (op == SIGMOID_GRAD) {
            r = y * x * (1.0f - x);
        } else if (op == TANH_GRAD) {
            r = y * (1.0f - x * x);
        } else if (op == GELU_GRAD) {
            r = y * (normalCdf(x) + x * INV_SQRT_2PI * TornadoMath.exp(-0.5f * x * x));
        } else if (op == GELU_TANH_GRAD) {
            float t = tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
            r = y * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_CUBIC * x * x));
        } else if (op == SILU_GRAD) {
            float s = sigmoid(x);
            r = y * s * (1.0f + x * (1.0f - s));
        } else if (op == LEAKY_RELU_GRAD) {
            r = x > 0.0f ? y : alpha * y;
        } else if (op == SOFTPLUS_GRAD) {
            r = y * sigmoid(x);
        } else if (op == BCE) {
            r = -(y * clampedLog(x) + (1.0f - y) * clampedLog(1.0f - x));
        } else if (op == BCE_DX) {
            float d = x * (1.0f - x);
            r = (x - y) / (d > 1e-12f ? d : 1e-12f);
        } else if (op == BCE_DY) {
            r = clampedLog(1.0f - x) - clampedLog(x);
        } else if (op == BCE_LOGITS) {
            r = (1.0f - y) * x + (x < 0.0f ? -x : 0.0f) + log1p(TornadoMath.exp(-TornadoMath.abs(x)));
        } else if (op == BCE_LOGITS_DX) {
            r = sigmoid(x) - y;
        } else if (op == ADD_RELU) {
            float s = x + y;
            r = s > 0.0f ? s : 0.0f;
        } else {
            r = x;
        }
        return r;
    }

    /** More activation gradients and PReLU (split out to stay under TornadoVM's inlining limit). */
    public static float binaryActivationGrad(int op, float x, float y, float alpha, float beta) {
        float r;
        if (op == ELU_GRAD) {
            r = x > 0.0f ? y : y * alpha * TornadoMath.exp(x);
        } else if (op == SELU_GRAD) {
            r = x > 0.0f ? y * SELU_SCALE : y * SELU_SCALE * SELU_ALPHA * TornadoMath.exp(x);
        } else if (op == CELU_GRAD) {
            r = x > 0.0f ? y : y * TornadoMath.exp(x / alpha);
        } else if (op == MISH_GRAD) {
            float t = tanh(softplus(x));
            r = y * (t + x * sigmoid(x) * (1.0f - t * t));
        } else if (op == SOFTSIGN_GRAD) {
            float d = 1.0f + TornadoMath.abs(x);
            r = y / (d * d);
        } else if (op == HARD_SIGMOID_GRAD) {
            r = (x > -3.0f) & (x < 3.0f) ? y / 6.0f : 0.0f;
        } else if (op == HARD_SWISH_GRAD) {
            r = x < -3.0f ? 0.0f : (x > 3.0f ? y : y * (2.0f * x + 3.0f) / 6.0f);
        } else if (op == THRESHOLD_GRAD) {
            r = x > alpha ? y : 0.0f;
        } else if (op == PRELU) {
            r = x > 0.0f ? x : y * x;
        } else if (op == PRELU_SLOPE) {
            r = x > 0.0f ? 1.0f : y;
        } else {
            r = x;
        }
        return r;
    }

    /** atan2, comparisons and logic (split out to stay under TornadoVM's inlining limit). */
    public static float binaryMath(int op, float x, float y, float alpha, float beta) {
        float r;
        if (op == ATAN2) {
            r = TornadoMath.atan2(x, y);
        } else if (op == NE) {
            r = x != y ? 1.0f : 0.0f;
        } else if (op == LE) {
            r = x <= y ? 1.0f : 0.0f;
        } else if (op == LOGICAL_AND) {
            r = (x != 0.0f) & (y != 0.0f) ? 1.0f : 0.0f;
        } else if (op == LOGICAL_OR) {
            r = (x != 0.0f) | (y != 0.0f) ? 1.0f : 0.0f;
        } else if (op == LOGICAL_XOR) {
            r = (x != 0.0f) != (y != 0.0f) ? 1.0f : 0.0f;
        } else {
            r = x;
        }
        return r;
    }

    /**
     * Unary op {@code op}. Kernels branch to the three groups themselves (see {@link #unary}): one
     * method holding them all exceeds TornadoVM's inlining limit of 600 nodes.
     */
    public static float unaryOp(int op, float x, float alpha, float beta) {
        float r;
        if (op >= IS_NAN) {
            r = unaryNaN(op, x, alpha, beta);
        } else if (op >= SIN) {
            r = unaryMath(op, x, alpha, beta);
        } else if (op >= ELU) {
            r = unaryActivation(op, x, alpha, beta);
        } else {
            r = unaryBase(op, x, alpha, beta);
        }
        return r;
    }

    /** Unary ops of waves 1-4. */
    public static float unaryBase(int op, float x, float alpha, float beta) {
        float r;
        if (op == NEG) {
            r = -x;
        } else if (op == EXP) {
            r = TornadoMath.exp(x);
        } else if (op == LOG) {
            r = TornadoMath.log(x);
        } else if (op == SQRT) {
            r = TornadoMath.sqrt(x);
        } else if (op == ABS) {
            r = TornadoMath.abs(x);
        } else if (op == RELU) {
            r = x > 0.0f ? x : 0.0f;
        } else if (op == AFFINE) {
            r = alpha * x + beta;
        } else if (op == POW_SCALAR) {
            r = TornadoMath.pow(x, alpha);
        } else if (op == CLAMP) {
            r = x < alpha ? alpha : (x > beta ? beta : x);
        } else if (op == RECIPROCAL) {
            r = 1.0f / x;
        } else if (op == SQUARE) {
            r = x * x;
        } else if (op == TANH) {
            r = tanh(x);
        } else if (op == SIGMOID) {
            r = sigmoid(x);
        } else if (op == GELU) {
            r = x * normalCdf(x);
        } else if (op == GELU_TANH) {
            r = 0.5f * x * (1.0f + tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)));
        } else if (op == SILU) {
            r = x * sigmoid(x);
        } else if (op == LEAKY_RELU) {
            r = x > 0.0f ? x : alpha * x;
        } else if (op == SOFTPLUS) {
            r = softplus(x);
        } else {
            r = alpha;
        }
        return r;
    }

    /** More activations (split out to stay under TornadoVM's inlining limit). */
    public static float unaryActivation(int op, float x, float alpha, float beta) {
        float r;
        if (op == ELU) {
            r = x > 0.0f ? x : alpha * expm1(x);
        } else if (op == SELU) {
            r = x > 0.0f ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * expm1(x);
        } else if (op == CELU) {
            r = x > 0.0f ? x : alpha * expm1(x / alpha);
        } else if (op == MISH) {
            r = x * tanh(softplus(x));
        } else if (op == SOFTSIGN) {
            r = x / (1.0f + TornadoMath.abs(x));
        } else if (op == HARD_SIGMOID) {
            float h = x / 6.0f + 0.5f;
            r = h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h);
        } else if (op == HARD_SWISH) {
            float h = x / 6.0f + 0.5f;
            r = x * (h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h));
        } else if (op == THRESHOLD) {
            r = x > alpha ? x : 0.0f;
        } else {
            r = x;
        }
        return r;
    }

    /** Trigonometric, hyperbolic, rounding and logic functions (split out to stay under TornadoVM's inlining limit). */
    public static float unaryMath(int op, float x, float alpha, float beta) {
        float r;
        if (op == SIN) {
            r = TornadoMath.sin(x);
        } else if (op == COS) {
            r = TornadoMath.cos(x);
        } else if (op == TAN) {
            r = TornadoMath.tan(x);
        } else if (op == ASIN) {
            r = TornadoMath.asin(x);
        } else if (op == ACOS) {
            r = TornadoMath.acos(x);
        } else if (op == ATAN) {
            r = TornadoMath.atan(x);
        } else if (op == SINH) {
            r = 0.5f * (expm1(x) - expm1(-x));
        } else if (op == COSH) {
            r = 0.5f * (TornadoMath.exp(x) + TornadoMath.exp(-x));
        } else if (op == ASINH) {
            r = TornadoMath.asinh(x);
        } else if (op == ACOSH) {
            r = TornadoMath.acosh(x);
        } else if (op == ATANH) {
            r = 0.5f * log1p(2.0f * x / (1.0f - x));
        } else if (op == ERF) {
            r = erf(x);
        } else if (op == ERFC) {
            r = erfc(x);
        } else if (op == ROUND) {
            r = roundHalfEven(x);
        } else if (op == TRUNC) {
            r = x < 0.0f ? TornadoMath.ceil(x) : TornadoMath.floor(x);
        } else if (op == FLOOR) {
            r = TornadoMath.floor(x);
        } else if (op == CEIL) {
            r = TornadoMath.ceil(x);
        } else if (op == SIGN) {
            r = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        } else if (op == LOG1P) {
            r = log1p(x);
        } else if (op == EXPM1) {
            r = expm1(x);
        } else if (op == LOGICAL_NOT) {
            r = x == 0.0f ? 1.0f : 0.0f;
        } else {
            r = x;
        }
        return r;
    }

    /** Same-shape binary op, no index arithmetic. */
    public static void binary(KernelContext kc, FloatArray a, FloatArray b, FloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i);
            float r;
            if (op >= MAXIMUM) {
                r = binaryNaN(op, x, y, alpha, beta);
            } else if (op >= ATAN2) {
                r = binaryMath(op, x, y, alpha, beta);
            } else if (op >= ELU_GRAD) {
                r = binaryActivationGrad(op, x, y, alpha, beta);
            } else {
                r = binaryBase(op, x, y, alpha, beta);
            }
            out.set(i, r);
        }
    }

    /**
     * Broadcasting binary op. {@code meta} = [rank, outShape[rank], aStrides[rank], bStrides[rank]]
     * where broadcast dimensions have stride 0.
     */
    public static void binaryBroadcast(KernelContext kc, FloatArray a, FloatArray b, FloatArray out, IntArray meta, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int rank = meta.get(0);
            int rem = i;
            int ai = 0;
            int bi = 0;
            for (int d = rank - 1; d >= 0; d--) {
                int size = meta.get(1 + d);
                int c = rem % size;
                rem = rem / size;
                ai += c * meta.get(1 + rank + d);
                bi += c * meta.get(1 + 2 * rank + d);
            }
            float x = a.get(ai);
            float y = b.get(bi);
            float r;
            if (op >= MAXIMUM) {
                r = binaryNaN(op, x, y, alpha, beta);
            } else if (op >= ATAN2) {
                r = binaryMath(op, x, y, alpha, beta);
            } else if (op >= ELU_GRAD) {
                r = binaryActivationGrad(op, x, y, alpha, beta);
            } else {
                r = binaryBase(op, x, y, alpha, beta);
            }
            out.set(i, r);
        }
    }

    /**
     * Binary op where each operand repeats one contiguous block of the output: element i reads
     * a[(i / aDiv) % aMod] and b[(i / bDiv) % bMod] (a full operand: div 1, mod n; a suffix such as
     * a bias: div 1; a per-channel [C, 1, 1]: div H W, mod C). No per-dimension index arithmetic.
     */
    public static void binaryRepeat(KernelContext kc, FloatArray a, FloatArray b, FloatArray out, int op, float alpha, float beta, int n, int aDiv, int aMod, int bDiv, int bMod) {
        int i = kc.globalIdx;
        if (i < n) {
            float x = a.get((i / aDiv) % aMod);
            float y = b.get((i / bDiv) % bMod);
            float r;
            if (op >= MAXIMUM) {
                r = binaryNaN(op, x, y, alpha, beta);
            } else if (op >= ATAN2) {
                r = binaryMath(op, x, y, alpha, beta);
            } else if (op >= ELU_GRAD) {
                r = binaryActivationGrad(op, x, y, alpha, beta);
            } else {
                r = binaryBase(op, x, y, alpha, beta);
            }
            out.set(i, r);
        }
    }

    public static void unary(KernelContext kc, FloatArray x, FloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float v = x.get(i);
            float r;
            if (op >= IS_NAN) {
                r = unaryNaN(op, v, alpha, beta);
            } else if (op >= SIN) {
                r = unaryMath(op, v, alpha, beta);
            } else if (op >= ELU) {
                r = unaryActivation(op, v, alpha, beta);
            } else {
                r = unaryBase(op, v, alpha, beta);
            }
            out.set(i, r);
        }
    }

    /** {@link #unary} on FP16 x and out (computed in FP32). */
    public static void unaryHalf(KernelContext kc, HalfFloatArray x, HalfFloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float v = x.get(i).getFloat32();
            float r;
            if (op >= IS_NAN) {
                r = unaryNaN(op, v, alpha, beta);
            } else if (op >= SIN) {
                r = unaryMath(op, v, alpha, beta);
            } else if (op >= ELU) {
                r = unaryActivation(op, v, alpha, beta);
            } else {
                r = unaryBase(op, v, alpha, beta);
            }
            out.set(i, new HalfFloat(r));
        }
    }

    /** {@link #binary} on FP16 a, b and out of the same shape (computed in FP32). */
    public static void binaryHalf(KernelContext kc, HalfFloatArray a, HalfFloatArray b, HalfFloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float x = a.get(i).getFloat32();
            float y = b.get(i).getFloat32();
            float r;
            if (op >= MAXIMUM) {
                r = binaryNaN(op, x, y, alpha, beta);
            } else if (op >= ATAN2) {
                r = binaryMath(op, x, y, alpha, beta);
            } else if (op >= ELU_GRAD) {
                r = binaryActivationGrad(op, x, y, alpha, beta);
            } else {
                r = binaryBase(op, x, y, alpha, beta);
            }
            out.set(i, new HalfFloat(r));
        }
    }

    /** {@link #binary} on an FP16 a and an FP32 b of the same shape, FP32 out (FP16 activations with FP32 gradients). */
    public static void binaryHalfA(KernelContext kc, HalfFloatArray a, FloatArray b, FloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float x = a.get(i).getFloat32();
            float y = b.get(i);
            float r;
            if (op >= MAXIMUM) {
                r = binaryNaN(op, x, y, alpha, beta);
            } else if (op >= ATAN2) {
                r = binaryMath(op, x, y, alpha, beta);
            } else if (op >= ELU_GRAD) {
                r = binaryActivationGrad(op, x, y, alpha, beta);
            } else {
                r = binaryBase(op, x, y, alpha, beta);
            }
            out.set(i, r);
        }
    }

    /** {@link #binary} on an FP32 a and an FP16 b of the same shape, FP32 out. */
    public static void binaryHalfB(KernelContext kc, FloatArray a, HalfFloatArray b, FloatArray out, int op, float alpha, float beta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float x = a.get(i);
            float y = b.get(i).getFloat32();
            float r;
            if (op >= MAXIMUM) {
                r = binaryNaN(op, x, y, alpha, beta);
            } else if (op >= ATAN2) {
                r = binaryMath(op, x, y, alpha, beta);
            } else if (op >= ELU_GRAD) {
                r = binaryActivationGrad(op, x, y, alpha, beta);
            } else {
                r = binaryBase(op, x, y, alpha, beta);
            }
            out.set(i, r);
        }
    }

    /** out = c != 0 ? a : b, all of n elements (broadcast beforehand). */
    public static void where(KernelContext kc, FloatArray c, FloatArray a, FloatArray b, FloatArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, c.get(i) != 0.0f ? a.get(i) : b.get(i));
        }
    }

    /** out[i, j] = labels[i] == j ? 1 : 0 for j &lt; classes; n = labels * classes. */
    public static void oneHot(KernelContext kc, IntArray labels, FloatArray out, int classes, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int row = i / classes;
            out.set(i, labels.get(row) == i - row * classes ? 1.0f : 0.0f);
        }
    }

    public static void fill(KernelContext kc, FloatArray out, float value, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, value);
        }
    }

    /**
     * Rotary position embedding: each pair (a, b) of a row at position t = (row / group) % T +
     * offset is rotated by angle t * freq_i, using the tables cos/sin [maxT, D/2]; group = 1 for
     * x [..., T, D] and H for x [..., T, H, D].
     * Pairs are (i, i + D/2) when {@code interleaved} is 0 (rotate-half) or (2i, 2i + 1)
     * otherwise. {@code sign} = -1 rotates backwards (the gradient). n = numel / 2.
     */
    public static void rope(KernelContext kc, FloatArray x, FloatArray out, FloatArray cos, FloatArray sin, int seq, int dim, int group, int offset, int interleaved, float sign, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int half = dim / 2;
            int row = i / half;
            int p = i - row * half;
            int t = (row / group) % seq + offset;
            int ia;
            int ib;
            if (interleaved != 0) {
                ia = row * dim + 2 * p;
                ib = ia + 1;
            } else {
                ia = row * dim + p;
                ib = ia + half;
            }
            float c = cos.get(t * half + p);
            float s = sign * sin.get(t * half + p);
            float a = x.get(ia);
            float b = x.get(ib);
            out.set(ia, a * c - b * s);
            out.set(ib, a * s + b * c);
        }
    }

    /** out[t, d] = table[index[t], d] for rows of {@code dim} elements; n = tokens * dim. */
    public static void gatherRows(KernelContext kc, FloatArray table, IntArray index, FloatArray out, int dim, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int t = i / dim;
            int d = i - t * dim;
            out.set(i, table.get(index.get(t) * dim + d));
        }
    }

    /**
     * Deterministic segment sum for the embedding gradient: out[v, d] = sum of dy[positions[k], d]
     * for k in offsets[v] .. offsets[v + 1] - 1, in order; n = rows * dim.
     */
    public static void segmentSumRows(KernelContext kc, FloatArray dy, IntArray offsets, IntArray positions, FloatArray out, int dim, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int v = i / dim;
            int d = i - v * dim;
            float acc = 0.0f;
            int end = offsets.get(v + 1);
            for (int k = offsets.get(v); k < end; k++) {
                acc += dy.get(positions.get(k) * dim + d);
            }
            out.set(i, acc);
        }
    }

    /**
     * Copies a block along one axis, with both tensors seen as [outer, length, inner]:
     * out[o, outOff + j, t] = x[o, inOff + j, t] for j &lt; len. {@code n} = outer * len * inner.
     */
    public static void copyBlock(KernelContext kc, FloatArray x, FloatArray out, int len, int inner, int inLen, int outLen, int inOff, int outOff, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int t = i % inner;
            int r = i / inner;
            int j = r % len;
            int o = r / len;
            out.set((o * outLen + outOff + j) * inner + t, x.get((o * inLen + inOff + j) * inner + t));
        }
    }

    /**
     * Gathers {@code x} into {@code out} with permuted strides. {@code meta} = [rank,
     * outShape[rank], inStride[rank]] where inStride[d] is the input stride of output dimension d.
     */
    public static void permute(KernelContext kc, FloatArray x, FloatArray out, IntArray meta, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int rank = meta.get(0);
            int rem = i;
            int src = 0;
            for (int d = rank - 1; d >= 0; d--) {
                int size = meta.get(1 + d);
                int c = rem % size;
                rem = rem / size;
                src += c * meta.get(1 + rank + d);
            }
            out.set(i, x.get(src));
        }
    }

    /** Tile edge of {@link #transposeTiled}; the local tile is padded to TT + 1 columns. */
    public static final int TT = 16;

    /**
     * Batched matrix transpose out[b, c, r] = x[b, r, c] through a padded local-memory tile, so
     * both the reads and the writes are coalesced. 16x16 threads per tile; groups cover
     * ceil(cols/16) x ceil(rows/16) tiles per batch, with the batch folded into the y dimension.
     */
    public static void transposeTiled(KernelContext kc, FloatArray x, FloatArray out, int rows, int cols, int tilesY) {
        float[] tile = kc.allocateFloatLocalArray(TT * (TT + 1));
        int tx = kc.localIdx;
        int ty = kc.localIdy;
        int gy = kc.groupIdy;
        int batch = gy / tilesY;
        int tileRow = gy - batch * tilesY;
        int r = tileRow * TT + ty;
        int c = kc.groupIdx * TT + tx;
        int base = batch * rows * cols;
        if ((r < rows) & (c < cols)) {
            tile[ty * (TT + 1) + tx] = x.get(base + r * cols + c);
        }
        kc.localBarrier();
        int orow = kc.groupIdx * TT + ty;
        int ocol = tileRow * TT + tx;
        if ((orow < cols) & (ocol < rows)) {
            out.set(base + orow * rows + ocol, tile[tx * (TT + 1) + ty]);
        }
    }

    /**
     * Batched transpose with 32x32 tiles and 32x8 threads, each moving four rows (the classic
     * shape for 32-wide SIMD groups). Groups cover ceil(cols/32) x ceil(rows/32) * batch tiles.
     */
    public static void transposeTiled32(KernelContext kc, FloatArray x, FloatArray out, int rows, int cols, int tilesY) {
        float[] tile = kc.allocateFloatLocalArray(32 * 33);
        int tx = kc.localIdx;
        int ty = kc.localIdy;
        int gy = kc.groupIdy;
        int batch = gy / tilesY;
        int tileRow = gy - batch * tilesY;
        int base = batch * rows * cols;
        int c = kc.groupIdx * 32 + tx;
        for (int j = 0; j < 32; j += 8) {
            int r = tileRow * 32 + ty + j;
            if ((r < rows) & (c < cols)) {
                tile[(ty + j) * 33 + tx] = x.get(base + r * cols + c);
            }
        }
        kc.localBarrier();
        int ocol = tileRow * 32 + tx;
        for (int j = 0; j < 32; j += 8) {
            int orow = kc.groupIdx * 32 + ty + j;
            if ((orow < cols) & (ocol < rows)) {
                out.set(base + orow * rows + ocol, tile[tx * 33 + ty + j]);
            }
        }
    }
}
