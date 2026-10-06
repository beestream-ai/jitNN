package org.beehive.jitnn;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.engine.Storage;
import org.beehive.jitnn.ops.Binary;
import org.beehive.jitnn.ops.Cast;
import org.beehive.jitnn.ops.MatMul;
import org.beehive.jitnn.ops.Permute;
import org.beehive.jitnn.ops.RandomFill;
import org.beehive.jitnn.ops.Reduce;
import org.beehive.jitnn.ops.Reshape;
import org.beehive.jitnn.ops.Sampling;
import org.beehive.jitnn.ops.Slice;
import org.beehive.jitnn.ops.Softmax;
import org.beehive.jitnn.ops.Unary;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * An n-dimensional array. Operations are lazy on TornadoVM: they are recorded and run at the next
 * sync point ({@link #item()}, {@link #toFloatArray()}, {@link Engine#sync}). Tensors are always
 * contiguous and row-major; {@link #reshape} shares storage, {@link #permute} copies.
 */
public final class Tensor {

    private final Shape shape;
    private final Storage storage;
    private Node gradFn;
    private boolean requiresGrad;
    private Tensor grad;

    private Tensor(Shape shape, Storage storage) {
        if (shape.numel() != storage.numel()) {
            throw new IllegalArgumentException("shape " + shape + " does not match storage of " + storage.numel() + " elements");
        }
        this.shape = shape;
        this.storage = storage;
    }

    // ---------------------------------------------------------------- creation

    public static Tensor of(float[] data, int... dims) {
        Shape s = dims.length == 0 && data.length != 1 ? Shape.of(data.length) : Shape.of(dims);
        if (s.numel() != data.length) {
            throw new IllegalArgumentException(data.length + " values for shape " + s);
        }
        return new Tensor(s, Storage.ofHost(FloatArray.fromArray(data.length == 0 ? new float[1] : data), DType.F32, false));
    }

    public static Tensor of(int[] data, int... dims) {
        Shape s = dims.length == 0 && data.length != 1 ? Shape.of(data.length) : Shape.of(dims);
        if (s.numel() != data.length) {
            throw new IllegalArgumentException(data.length + " values for shape " + s);
        }
        return new Tensor(s, Storage.ofHost(IntArray.fromArray(data.length == 0 ? new int[1] : data), DType.I32, false));
    }

    public static Tensor scalar(float value) {
        return new Tensor(Shape.scalar(), Storage.ofHost(FloatArray.fromElements(value), DType.F32, false));
    }

    public static Tensor full(float value, int... dims) {
        Shape s = Shape.of(dims);
        FloatArray a = new FloatArray(Math.max(s.numel(), 1));
        a.init(value);
        return new Tensor(s, Storage.ofHost(a, DType.F32, false));
    }

    public static Tensor zeros(int... dims) {
        return full(0.0f, dims);
    }

    public static Tensor ones(int... dims) {
        return full(1.0f, dims);
    }

    /** Uniform in [lo, hi), from a counter-based generator: the same seed gives the same values everywhere. */
    public static Tensor uniform(long seed, float lo, float hi, int... dims) {
        return RandomFill.apply(Shape.of(dims), seed, 0, lo, hi);
    }

    /**
     * As {@link #uniform(long, float, float, int...)} with the seed in an INT32 [2] tensor (see
     * {@link #seed}): a plan replayed with a new seed draws new values without recompiling, so use
     * this for noise drawn every step.
     */
    public static Tensor uniform(Tensor seed, float lo, float hi, int... dims) {
        return RandomFill.apply(Shape.of(dims), seed, 0, lo, hi);
    }

    /** As {@link #normal(long, float, float, int...)} with the seed in an INT32 [2] tensor. */
    public static Tensor normal(Tensor seed, float mean, float std, int... dims) {
        return RandomFill.apply(Shape.of(dims), seed, 1, mean, std);
    }

    /** A 64-bit seed as the INT32 [2] host tensor [lo, hi] that the seeded random operators take. */
    public static Tensor seed(long seed) {
        return of(new int[] { (int) seed, (int) (seed >> 32) }, 2);
    }

    /** Normal with the given mean and standard deviation, counter-based like {@link #uniform}. */
    public static Tensor normal(long seed, float mean, float std, int... dims) {
        return RandomFill.apply(Shape.of(dims), seed, 1, mean, std);
    }

    /**
     * A trainable parameter: persistent storage (it stays on the device across steps), gradient
     * tracking on, initialised with the values of {@code init}.
     */
    public static Tensor parameter(Tensor init) {
        Tensor p = persistent(init);
        p.requiresGrad = true;
        return p;
    }

    /** Persistent storage without gradient tracking (optimiser state, running statistics). */
    public static Tensor persistent(Tensor init) {
        float[] values = init.toFloatArray();
        Tensor p = new Tensor(init.shape, Storage.ofHost(FloatArray.fromArray(values.length == 0 ? new float[1] : values), DType.F32, true));
        return p;
    }

    // ---------------------------------------------------------------- internal construction

    /** An output tensor for an operator; its storage is allocated when the operator runs. */
    public static Tensor result(Shape shape, DType dtype) {
        return new Tensor(shape, new Storage(dtype, shape.numel(), false));
    }

    /** A tensor that shares {@code base}'s storage under another shape. */
    public static Tensor alias(Tensor base, Shape shape) {
        return new Tensor(shape, base.storage);
    }

    /**
     * Records an operator application. Outputs track gradients when gradient mode is on and any
     * input does; the node is traced (run at the next sync) unless {@code traced} is false.
     */
    public static void record(Op op, Tensor[] inputs, Tensor[] outputs, int[] iattr, float[] fattr, boolean traced) {
        Node node = new Node(op, inputs, outputs, iattr, fattr, traced);
        if (Grad.isEnabled()) {
            boolean needs = false;
            for (Tensor t : inputs) {
                needs |= t.requiresGrad;
            }
            if (needs) {
                for (Tensor t : outputs) {
                    // a view of a parameter (untraced reshape) shares its persistent storage but is a new tensor
                    if (!t.storage.isPersistent() || !traced) {
                        t.requiresGrad = true;
                        t.gradFn = node;
                    }
                }
            }
        }
        if (traced) {
            Engine.record(node);
        }
    }

    public Storage storage() {
        return storage;
    }

    public Node gradFn() {
        return gradFn;
    }

    void accumulateGrad(Tensor g) {
        grad = grad == null ? g : grad.add(g);
    }

    void setGrad(Tensor g) {
        grad = g;
    }

    // ---------------------------------------------------------------- metadata

    public Shape shape() {
        return shape;
    }

    public DType dtype() {
        return storage.dtype();
    }

    public int numel() {
        return shape.numel();
    }

    public int rank() {
        return shape.rank();
    }

    public int dim(int axis) {
        return shape.dim(axis);
    }

    // ---------------------------------------------------------------- autograd

    public boolean requiresGrad() {
        return requiresGrad;
    }

    /** Turns gradient tracking on or off for a leaf tensor. */
    public Tensor requiresGrad(boolean on) {
        if (gradFn != null) {
            throw new IllegalStateException("requiresGrad can only be set on leaf tensors");
        }
        requiresGrad = on;
        return this;
    }

    /** The accumulated gradient of a leaf tensor after {@link #backward()}, or null. */
    public Tensor grad() {
        return grad;
    }

    public void zeroGrad() {
        grad = null;
    }

    /** Back-propagates from this scalar. */
    public void backward() {
        if (numel() != 1) {
            throw new IllegalStateException("backward() needs a scalar; pass a gradient for shape " + shape);
        }
        Autograd.backward(this, null);
    }

    public void backward(Tensor gradient) {
        Autograd.backward(this, gradient);
    }

    /** The same data without gradient history. */
    public Tensor detach() {
        return new Tensor(shape, storage);
    }

    // ---------------------------------------------------------------- reading

    public float item() {
        if (numel() != 1) {
            throw new IllegalStateException("item() needs a single element, shape is " + shape);
        }
        Engine.sync(this);
        return storage.dtype() == DType.F32 ? ((FloatArray) storage.host()).get(0) : toFloatArray()[0];
    }

    public float[] toFloatArray() {
        Engine.sync(this);
        if (storage.dtype() == DType.I32) {
            int[] v = toIntArray();
            float[] f = new float[v.length];
            for (int i = 0; i < v.length; i++) {
                f[i] = v[i];
            }
            return f;
        }
        if (storage.dtype() == DType.F16) {
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray h = (uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray) storage.host();
            float[] out = new float[numel()];
            for (int i = 0; i < out.length; i++) {
                out[i] = h.get(i).getFloat32();
            }
            return out;
        }
        if (storage.dtype() == DType.BF16) {
            uk.ac.manchester.tornado.api.types.arrays.ShortArray b = (uk.ac.manchester.tornado.api.types.arrays.ShortArray) storage.host();
            float[] out = new float[numel()];
            for (int i = 0; i < out.length; i++) {
                out[i] = Float.intBitsToFloat(((int) b.get(i)) << 16);
            }
            return out;
        }
        FloatArray a = (FloatArray) storage.host();
        float[] out = new float[numel()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i);
        }
        return out;
    }

    public int[] toIntArray() {
        Engine.sync(this);
        if (storage.dtype() != DType.I32) {
            throw new IllegalStateException("not an int tensor");
        }
        IntArray a = (IntArray) storage.host();
        int[] out = new int[numel()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i);
        }
        return out;
    }

    /** Overwrites a persistent tensor's contents from the host (e.g. loading weights). */
    public void copyFrom(float[] values) {
        if (values.length != numel() || storage.dtype() != DType.F32) {
            throw new IllegalArgumentException("expected " + numel() + " floats");
        }
        Engine.sync();
        FloatArray a = (FloatArray) storage.host();
        for (int i = 0; i < values.length; i++) {
            a.set(i, values[i]);
        }
        storage.hostWritten();
    }

    // ---------------------------------------------------------------- element-wise

    public Tensor add(Tensor o) {
        return Binary.apply(Binary.Kind.ADD, this, o);
    }

    public Tensor sub(Tensor o) {
        return Binary.apply(Binary.Kind.SUB, this, o);
    }

    public Tensor mul(Tensor o) {
        return Binary.apply(Binary.Kind.MUL, this, o);
    }

    public Tensor div(Tensor o) {
        return Binary.apply(Binary.Kind.DIV, this, o);
    }

    public Tensor maximum(Tensor o) {
        return Binary.apply(Binary.Kind.MAX, this, o);
    }

    public Tensor minimum(Tensor o) {
        return Binary.apply(Binary.Kind.MIN, this, o);
    }

    public Tensor gt(Tensor o) {
        return Binary.apply(Binary.Kind.GT, this, o);
    }

    public Tensor lt(Tensor o) {
        return Binary.apply(Binary.Kind.LT, this, o);
    }

    public Tensor ge(Tensor o) {
        return Binary.apply(Binary.Kind.GE, this, o);
    }

    public Tensor eq(Tensor o) {
        return Binary.apply(Binary.Kind.EQ, this, o);
    }

    public Tensor add(float c) {
        return Unary.apply(Unary.Kind.AFFINE, this, 1.0f, c);
    }

    public Tensor mul(float c) {
        return Unary.apply(Unary.Kind.AFFINE, this, c, 0.0f);
    }

    public Tensor neg() {
        return Unary.apply(Unary.Kind.NEG, this, 0, 0);
    }

    public Tensor exp() {
        return Unary.apply(Unary.Kind.EXP, this, 0, 0);
    }

    public Tensor log() {
        return Unary.apply(Unary.Kind.LOG, this, 0, 0);
    }

    public Tensor sqrt() {
        return Unary.apply(Unary.Kind.SQRT, this, 0, 0);
    }

    public Tensor abs() {
        return Unary.apply(Unary.Kind.ABS, this, 0, 0);
    }

    public Tensor relu() {
        return Unary.apply(Unary.Kind.RELU, this, 0, 0);
    }

    /** min(max(x, 0), 6). */
    public Tensor relu6() {
        return clamp(0.0f, 6.0f);
    }

    public Tensor pow(float exponent) {
        return Unary.apply(Unary.Kind.POW_SCALAR, this, exponent, 0);
    }

    public Tensor clamp(float lo, float hi) {
        return Unary.apply(Unary.Kind.CLAMP, this, lo, hi);
    }

    public Tensor reciprocal() {
        return Unary.apply(Unary.Kind.RECIPROCAL, this, 0, 0);
    }

    public Tensor square() {
        return Unary.apply(Unary.Kind.SQUARE, this, 0, 0);
    }

    public Tensor tanh() {
        return Unary.apply(Unary.Kind.TANH, this, 0, 0);
    }

    public Tensor sigmoid() {
        return Unary.apply(Unary.Kind.SIGMOID, this, 0, 0);
    }

    /** GELU, exact form x * Phi(x) (PyTorch's default, approximate='none'). */
    public Tensor gelu() {
        return Unary.apply(Unary.Kind.GELU, this, 0, 0);
    }

    /** GELU with the tanh approximation (GPT-2's form, PyTorch's approximate='tanh'). */
    public Tensor geluTanh() {
        return Unary.apply(Unary.Kind.GELU_TANH, this, 0, 0);
    }

    /** SiLU (swish): x * sigmoid(x). */
    public Tensor silu() {
        return Unary.apply(Unary.Kind.SILU, this, 0, 0);
    }

    /** LeakyReLU: x for x > 0, else {@code slope} * x (PyTorch's default slope is 0.01). */
    public Tensor leakyRelu(float slope) {
        return Unary.apply(Unary.Kind.LEAKY_RELU, this, slope, 0);
    }

    /** Element-wise sine. */
    public Tensor sin() {
        return Unary.apply(Unary.Kind.SIN, this, 0, 0);
    }

    /** Element-wise cosine. */
    public Tensor cos() {
        return Unary.apply(Unary.Kind.COS, this, 0, 0);
    }

    /** Element-wise tangent. */
    public Tensor tan() {
        return Unary.apply(Unary.Kind.TAN, this, 0, 0);
    }

    /** Element-wise arcsine. */
    public Tensor asin() {
        return Unary.apply(Unary.Kind.ASIN, this, 0, 0);
    }

    /** Element-wise arccosine. */
    public Tensor acos() {
        return Unary.apply(Unary.Kind.ACOS, this, 0, 0);
    }

    /** Element-wise arctangent. */
    public Tensor atan() {
        return Unary.apply(Unary.Kind.ATAN, this, 0, 0);
    }

    /** Element-wise hyperbolic sine. */
    public Tensor sinh() {
        return Unary.apply(Unary.Kind.SINH, this, 0, 0);
    }

    /** Element-wise hyperbolic cosine. */
    public Tensor cosh() {
        return Unary.apply(Unary.Kind.COSH, this, 0, 0);
    }

    /** Element-wise inverse hyperbolic sine. */
    public Tensor asinh() {
        return Unary.apply(Unary.Kind.ASINH, this, 0, 0);
    }

    /** Element-wise inverse hyperbolic cosine. */
    public Tensor acosh() {
        return Unary.apply(Unary.Kind.ACOSH, this, 0, 0);
    }

    /** Element-wise inverse hyperbolic tangent. */
    public Tensor atanh() {
        return Unary.apply(Unary.Kind.ATANH, this, 0, 0);
    }

    /** Element-wise error function. */
    public Tensor erf() {
        return Unary.apply(Unary.Kind.ERF, this, 0, 0);
    }

    /** Element-wise complementary error function. */
    public Tensor erfc() {
        return Unary.apply(Unary.Kind.ERFC, this, 0, 0);
    }

    /** Element-wise rounding to the nearest integer, halves to even (PyTorch's round). */
    public Tensor round() {
        return Unary.apply(Unary.Kind.ROUND, this, 0, 0);
    }

    /** Element-wise rounding towards zero. */
    public Tensor trunc() {
        return Unary.apply(Unary.Kind.TRUNC, this, 0, 0);
    }

    /** Element-wise floor. */
    public Tensor floor() {
        return Unary.apply(Unary.Kind.FLOOR, this, 0, 0);
    }

    /** Element-wise ceiling. */
    public Tensor ceil() {
        return Unary.apply(Unary.Kind.CEIL, this, 0, 0);
    }

    /** Element-wise sign: -1, 0 or 1. */
    public Tensor sign() {
        return Unary.apply(Unary.Kind.SIGN, this, 0, 0);
    }

    /** Element-wise log(1 + x), accurate near 0. */
    public Tensor log1p() {
        return Unary.apply(Unary.Kind.LOG1P, this, 0, 0);
    }

    /** Element-wise exp(x) - 1, accurate near 0. */
    public Tensor expm1() {
        return Unary.apply(Unary.Kind.EXPM1, this, 0, 0);
    }

    /** Element-wise logical not: 1 where x is 0, else 0. */
    public Tensor logicalNot() {
        return Unary.apply(Unary.Kind.LOGICAL_NOT, this, 0, 0);
    }

    /** atan2(this, o), the angle of the point (o, this), with broadcasting. */
    public Tensor atan2(Tensor o) {
        return Binary.apply(Binary.Kind.ATAN2, this, o);
    }

    /** 1 where this != o, else 0, with broadcasting. */
    public Tensor ne(Tensor o) {
        return Binary.apply(Binary.Kind.NE, this, o);
    }

    /** 1 where this <= o, else 0, with broadcasting. */
    public Tensor le(Tensor o) {
        return Binary.apply(Binary.Kind.LE, this, o);
    }

    /** 1 where both are non-zero, with broadcasting. */
    public Tensor logicalAnd(Tensor o) {
        return Binary.apply(Binary.Kind.LOGICAL_AND, this, o);
    }

    /** 1 where either is non-zero, with broadcasting. */
    public Tensor logicalOr(Tensor o) {
        return Binary.apply(Binary.Kind.LOGICAL_OR, this, o);
    }

    /** 1 where exactly one is non-zero, with broadcasting. */
    public Tensor logicalXor(Tensor o) {
        return Binary.apply(Binary.Kind.LOGICAL_XOR, this, o);
    }

    /** ELU: x for x > 0, else alpha (exp(x) - 1). */
    public Tensor elu(float alpha) {
        return Unary.apply(Unary.Kind.ELU, this, alpha, 0);
    }

    /** SELU with PyTorch's constants (alpha 1.6732632, scale 1.0507010). */
    public Tensor selu() {
        return Unary.apply(Unary.Kind.SELU, this, 0, 0);
    }

    /** CELU: max(0, x) + min(0, alpha (exp(x / alpha) - 1)). */
    public Tensor celu(float alpha) {
        return Unary.apply(Unary.Kind.CELU, this, alpha, 0);
    }

    /** Mish: x tanh(softplus(x)). */
    public Tensor mish() {
        return Unary.apply(Unary.Kind.MISH, this, 0, 0);
    }

    /** Softsign: x / (1 + |x|). */
    public Tensor softsign() {
        return Unary.apply(Unary.Kind.SOFTSIGN, this, 0, 0);
    }

    /** HardSigmoid, PyTorch's form: clamp(x / 6 + 1/2, 0, 1). */
    public Tensor hardSigmoid() {
        return Unary.apply(Unary.Kind.HARD_SIGMOID, this, 0, 0);
    }

    /** HardSwish: x hardSigmoid(x). */
    public Tensor hardSwish() {
        return Unary.apply(Unary.Kind.HARD_SWISH, this, 0, 0);
    }

    /** HardTanh: x clamped to [lo, hi] (PyTorch's default -1, 1). */
    public Tensor hardTanh(float lo, float hi) {
        return clamp(lo, hi);
    }

    /** ThresholdedReLU (Keras): x where x > theta, else 0. */
    public Tensor thresholdedRelu(float theta) {
        return Unary.apply(Unary.Kind.THRESHOLD, this, theta, 0);
    }

    /** x^3 (DL4J's Cube activation). */
    public Tensor cube() {
        return square().mul(this);
    }

    /** Softplus, log(1 + exp(x)) (beta 1), computed without overflow. */
    public Tensor softplus() {
        return Unary.apply(Unary.Kind.SOFTPLUS, this, 0, 0);
    }

    // ---------------------------------------------------------------- reductions

    public Tensor sum() {
        return reshape(numel()).sum(0, false);
    }

    public Tensor sum(int axis, boolean keepDim) {
        return Reduce.apply(Reduce.Kind.SUM, this, axis, keepDim);
    }

    public Tensor mean() {
        return reshape(numel()).mean(0, false);
    }

    public Tensor mean(int axis, boolean keepDim) {
        return Reduce.apply(Reduce.Kind.MEAN, this, axis, keepDim);
    }

    public Tensor max(int axis, boolean keepDim) {
        return Reduce.apply(Reduce.Kind.MAX, this, axis, keepDim);
    }

    public Tensor min(int axis, boolean keepDim) {
        return Reduce.apply(Reduce.Kind.MIN, this, axis, keepDim);
    }

    /** Index of the maximum along {@code axis}, as an INT32 tensor (no gradient). */
    public Tensor argmax(int axis) {
        return Reduce.argmax(this, axis);
    }

    /** Sums broadcast dimensions away so the result has {@code target} shape (gradient of broadcasting). */
    public Tensor sumTo(Shape target) {
        if (shape.equals(target)) {
            return this;
        }
        Tensor t = this;
        int extra = t.rank() - target.rank();
        for (int i = 0; i < extra; i++) {
            t = t.sum(0, false);
        }
        for (int d = 0; d < target.rank(); d++) {
            if (target.dim(d) == 1 && t.dim(d) != 1) {
                t = t.sum(d, true);
            }
        }
        return t.shape.equals(target) ? t : t.reshape(target.dims());
    }

    /** Broadcast copy to {@code target} shape. */
    public Tensor expand(Shape target) {
        if (shape.equals(target)) {
            return this;
        }
        return Binary.apply(Binary.Kind.FIRST, this, this, target);
    }

    // ---------------------------------------------------------------- shape

    /** Same data, new shape; one dimension may be -1. */
    public Tensor reshape(int... dims) {
        return Reshape.apply(this, dims);
    }

    public Tensor permute(int... order) {
        return Permute.apply(this, order);
    }

    public Tensor transpose(int a, int b) {
        int[] order = new int[rank()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        int x = shape.axis(a);
        int y = shape.axis(b);
        order[x] = y;
        order[y] = x;
        return permute(order);
    }

    /** This tensor converted to {@code dtype} (FP32, FP16 or BF16; round to nearest even); differentiable. */
    public Tensor to(DType dtype) {
        return Cast.apply(this, dtype);
    }

    public Tensor half() {
        return Cast.apply(this, DType.F16);
    }

    public Tensor bfloat16() {
        return Cast.apply(this, DType.BF16);
    }

    public Tensor toFloat() {
        return Cast.apply(this, DType.F32);
    }

    /** The k largest values along the last axis and their INT32 indices (descending; ties: lower index first). */
    public Tensor[] topk(int k) {
        return Sampling.topk(this, k);
    }

    /** Elements {@code start .. start + length - 1} along {@code axis}, as a new tensor. */
    public Tensor narrow(int axis, int start, int length) {
        return Slice.narrow(this, axis, start, length);
    }

    /** Elements {@code start .. end - 1} along {@code axis}; negative indices count from the end. */
    public Tensor slice(int axis, int start, int end) {
        int size = dim(axis);
        int s = start < 0 ? start + size : start;
        int e = end < 0 ? end + size : Math.min(end, size);
        return Slice.narrow(this, axis, s, Math.max(e - s, 0));
    }

    /** Splits {@code axis} into consecutive pieces of the given sizes. */
    public Tensor[] split(int axis, int... sizes) {
        Tensor[] parts = new Tensor[sizes.length];
        int start = 0;
        for (int i = 0; i < sizes.length; i++) {
            parts[i] = Slice.narrow(this, axis, start, sizes[i]);
            start += sizes[i];
        }
        if (start != dim(axis)) {
            throw new IllegalArgumentException("split sizes add up to " + start + ", axis " + axis + " of " + shape + " has " + dim(axis));
        }
        return parts;
    }

    /** Splits {@code axis} into {@code chunks} equal pieces. */
    public Tensor[] chunk(int chunks, int axis) {
        if (dim(axis) % chunks != 0) {
            throw new IllegalArgumentException("axis " + axis + " of " + shape + " does not split into " + chunks + " equal chunks");
        }
        int[] sizes = new int[chunks];
        java.util.Arrays.fill(sizes, dim(axis) / chunks);
        return split(axis, sizes);
    }

    /** Joins tensors along an existing axis. */
    public static Tensor cat(int axis, Tensor... parts) {
        return Slice.concat(axis, parts);
    }

    /** Joins same-shape tensors along a new axis. */
    public static Tensor stack(int axis, Tensor... parts) {
        Tensor[] expanded = new Tensor[parts.length];
        for (int i = 0; i < parts.length; i++) {
            expanded[i] = parts[i].unsqueeze(axis);
        }
        return Slice.concat(axis, expanded);
    }

    /** A view with a new size-1 axis at {@code axis} (negative counts from the end of the result). */
    public Tensor unsqueeze(int axis) {
        int r = rank() + 1;
        int ax = axis < 0 ? axis + r : axis;
        int[] dims = new int[r];
        for (int d = 0, s = 0; d < r; d++) {
            dims[d] = d == ax ? 1 : shape.dim(s++);
        }
        return reshape(dims);
    }

    /** A view without the size-1 axis {@code axis}. */
    public Tensor squeeze(int axis) {
        int ax = shape.axis(axis);
        if (dim(ax) != 1) {
            throw new IllegalArgumentException("axis " + axis + " of " + shape + " is not 1");
        }
        int[] dims = new int[rank() - 1];
        for (int d = 0, o = 0; d < rank(); d++) {
            if (d != ax) {
                dims[o++] = shape.dim(d);
            }
        }
        return reshape(dims);
    }

    /** Transpose of a matrix. */
    public Tensor t() {
        if (rank() != 2) {
            throw new IllegalStateException("t() needs a matrix, shape is " + shape);
        }
        return transpose(0, 1);
    }

    // ---------------------------------------------------------------- linear algebra

    public Tensor matmul(Tensor b) {
        return MatMul.apply(Autocast.operand(this), Autocast.operand(b), false, false);
    }

    /** op(this) * op(b), where op transposes when the flag is set, without materialising the transpose. */
    public Tensor matmul(Tensor b, boolean transposeThis, boolean transposeB) {
        return MatMul.apply(Autocast.operand(this), Autocast.operand(b), transposeThis, transposeB);
    }

    // ---------------------------------------------------------------- softmax

    /** Softmax along the last axis. */
    public Tensor softmax() {
        return Softmax.apply(this, false);
    }

    /** Log-softmax along the last axis. */
    public Tensor logSoftmax() {
        return Softmax.apply(this, true);
    }

    @Override
    public String toString() {
        return "Tensor" + shape + " " + dtype() + (requiresGrad ? " grad" : "");
    }
}
