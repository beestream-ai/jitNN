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
        return new Tensor(s, Storage.ofHost(floats(data.length == 0 ? new float[1] : data), DType.F32, false));
    }

    /**
     * A tensor over {@code data} itself, without a copy, as PyTorch's {@code from_numpy}: the array is
     * read when the operators that use it run (at the next sync), so it must not change until then,
     * and later reads of the tensor see later changes. {@link #of} copies instead.
     */
    public static Tensor wrap(float[] data, int... dims) {
        Shape s = dims.length == 0 && data.length != 1 ? Shape.of(data.length) : Shape.of(dims);
        if (s.numel() != data.length || data.length == 0) {
            throw new IllegalArgumentException(data.length + " values for shape " + s);
        }
        return new Tensor(s, Storage.ofHeap(data, DType.F32, data.length));
    }

    /** {@link #wrap(float[], int...)} for int data. */
    public static Tensor wrap(int[] data, int... dims) {
        Shape s = dims.length == 0 && data.length != 1 ? Shape.of(data.length) : Shape.of(dims);
        if (s.numel() != data.length || data.length == 0) {
            throw new IllegalArgumentException(data.length + " values for shape " + s);
        }
        return new Tensor(s, Storage.ofHeap(data, DType.I32, data.length));
    }

    public static Tensor of(int[] data, int... dims) {
        Shape s = dims.length == 0 && data.length != 1 ? Shape.of(data.length) : Shape.of(dims);
        if (s.numel() != data.length) {
            throw new IllegalArgumentException(data.length + " values for shape " + s);
        }
        return new Tensor(s, Storage.ofHost(ints(data.length == 0 ? new int[1] : data), DType.I32, false));
    }

    /**
     * A native copy of data in one bulk copy (TornadoVM's FloatArray.fromArray sets one element at a
     * time: 3.3 ms for a batch of 32 ImageNet images against well under 1 ms).
     */
    static FloatArray floats(float[] data) {
        FloatArray a = new FloatArray(data.length);
        java.lang.foreign.MemorySegment.copy(data, 0, a.getSegment(), java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, data.length);
        return a;
    }

    /** {@link #floats} for int data. */
    static IntArray ints(int[] data) {
        IntArray a = new IntArray(data.length);
        java.lang.foreign.MemorySegment.copy(data, 0, a.getSegment(), java.lang.foreign.ValueLayout.JAVA_INT, 0, data.length);
        return a;
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

    /** The n x m matrix with ones on diagonal k (0 main, positive above, negative below), zeros elsewhere. */
    public static Tensor eye(int n, int m, int k) {
        float[] v = new float[n * m];
        for (int i = 0; i < n; i++) {
            int j = i + k;
            if (j >= 0 && j < m) {
                v[i * m + j] = 1.0f;
            }
        }
        return of(v, n, m);
    }

    /** {@link #eye} with the shape of the rank-2 tensor {@code like} (ONNX EyeLike). */
    public static Tensor eyeLike(Tensor like, int k) {
        if (like.rank() != 2) {
            throw new IllegalArgumentException("eyeLike: needs a rank-2 tensor, got " + like.shape());
        }
        return eye(like.dim(0), like.dim(1), k);
    }

    /** start, start + step, ... below end (above end for a negative step): torch.arange, ONNX Range. */
    public static Tensor arange(float start, float end, float step) {
        if (step == 0) {
            throw new IllegalArgumentException("arange: step must not be 0");
        }
        int n = Math.max(0, (int) Math.ceil((end - start) / step));
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = start + i * step;
        }
        return of(v, n);
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
        // a random initial value just recorded is drawn on the host (same values), not by a device plan
        if (!init.storage.isHostValid() && backendIsDevice()) {
            org.beehive.jitnn.engine.Engine.generateOnHost(init);
        }
        // host data (not persistent, so no pending operator can write it) is copied without a sync,
        // which would run everything recorded so far
        boolean plainHost = init.storage.isHostValid() && !init.storage.isPersistent() && init.storage.dtype() == DType.F32 && init.storage.host() instanceof FloatArray;
        float[] values = plainHost ? java.util.Arrays.copyOf(((FloatArray) init.storage.host()).toHeapArray(), init.numel()) : init.toFloatArray();
        Tensor p = new Tensor(init.shape, Storage.ofHost(floats(values.length == 0 ? new float[1] : values), DType.F32, true));
        return p;
    }

    /**
     * A weight in one of GGUF's block-quantised formats (Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K) from its
     * raw blocks, as a GGUF file stores them: persistent, so it stays on the device. The last axis
     * must be a multiple of the format's block size.
     */
    public static Tensor quantized(DType type, byte[] blocks, int... dims) {
        Shape s = Shape.of(dims);
        if (!type.isBlockQuantized() || dims.length == 0 || dims[dims.length - 1] % type.blockValues() != 0) {
            throw new IllegalArgumentException(type + " weight of shape " + s + ": not a GGUF block format or last axis not a multiple of its block");
        }
        long bytes = (long) s.numel() / type.blockValues() * type.blockBytes();
        if (blocks.length != bytes) {
            throw new IllegalArgumentException(type + " weight of shape " + s + " needs " + bytes + " bytes, got " + blocks.length);
        }
        return new Tensor(s, Storage.ofHost(uk.ac.manchester.tornado.api.types.arrays.ByteArray.fromArray(blocks), type, s.numel(), true));
    }

    /** Persistent FP16 storage (weights kept in half precision on the device), from FP32 values rounded to nearest. */
    public static Tensor persistentHalf(float[] values, int... dims) {
        uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray h = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(Math.max(values.length, 1));
        for (int i = 0; i < values.length; i++) {
            h.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat(values[i]));
        }
        return new Tensor(Shape.of(dims), Storage.ofHost(h, DType.F16, true));
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
        if (traced && !op.acceptsHalf() && Autocast.halfActivations()) {
            // half-precision inference: an operator without an FP16 form reads FP32 copies
            for (int i = 0; i < inputs.length; i++) {
                if (inputs[i] != null && inputs[i].dtype() == DType.F16) {
                    inputs = inputs.clone();
                    inputs[i] = inputs[i].to(DType.F32);
                }
            }
        }
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

    private static boolean backendIsDevice() {
        return org.beehive.jitnn.engine.Engine.backend() == org.beehive.jitnn.engine.Engine.Backend.TORNADO;
    }

    /**
     * Declares a persistent tensor (a weight) no longer used, as when its model is closed:
     * compiled plans that use it can then run another model of the same structure without
     * compiling again. Using the tensor afterwards is an error.
     */
    public void release() {
        storage.release();
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
        if (storage.dtype() == DType.FP8_E4M3 || storage.dtype() == DType.FP8_E5M2) {
            uk.ac.manchester.tornado.api.types.arrays.ByteArray q = (uk.ac.manchester.tornado.api.types.arrays.ByteArray) storage.host();
            float[] out = new float[numel()];
            boolean e4m3 = storage.dtype() == DType.FP8_E4M3;
            for (int i = 0; i < out.length; i++) {
                out[i] = org.beehive.jitnn.kernels.Fp8Kernels.decode(q.get(i), e4m3 ? 0 : 1, Float.NaN, Float.POSITIVE_INFINITY);
            }
            return out;
        }
        if (storage.dtype().isBlockQuantized() && storage.dtype() != DType.Q8_0) {
            uk.ac.manchester.tornado.api.types.arrays.ByteArray q = (uk.ac.manchester.tornado.api.types.arrays.ByteArray) storage.host();
            float[] out = new float[numel()];
            for (int i = 0; i < out.length; i++) {
                out[i] = org.beehive.jitnn.kernels.GgufKernels.value(q, org.beehive.jitnn.kernels.GgufKernels.code(storage.dtype()), i);
            }
            return out;
        }
        if (storage.dtype() == DType.Q8_0) {
            uk.ac.manchester.tornado.api.types.arrays.ByteArray q = (uk.ac.manchester.tornado.api.types.arrays.ByteArray) storage.host();
            float[] out = new float[numel()];
            for (int i = 0; i < out.length; i++) {
                int block = i / 32 * 34;
                out[i] = q.getHalfFloat(block).getFloat32() * q.get(block + 2 + i % 32);
            }
            return out;
        }
        if (storage.dtype() == DType.F16) {
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray h = (uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray) storage.host();
            // the raw FP16 bits in one copy, converted without a HalfFloat object per element
            short[] bits = h.toShortArray();
            float[] out = new float[numel()];
            for (int i = 0; i < out.length; i++) {
                out[i] = Float.float16ToFloat(bits[i]);
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

    /** Bitwise and of INT32 tensors, with broadcasting. */
    public Tensor bitwiseAnd(Tensor o) {
        return org.beehive.jitnn.ops.Bitwise.apply(org.beehive.jitnn.ops.Bitwise.Kind.AND, this, o);
    }

    /** Bitwise or of INT32 tensors, with broadcasting. */
    public Tensor bitwiseOr(Tensor o) {
        return org.beehive.jitnn.ops.Bitwise.apply(org.beehive.jitnn.ops.Bitwise.Kind.OR, this, o);
    }

    /** Bitwise exclusive or of INT32 tensors, with broadcasting. */
    public Tensor bitwiseXor(Tensor o) {
        return org.beehive.jitnn.ops.Bitwise.apply(org.beehive.jitnn.ops.Bitwise.Kind.XOR, this, o);
    }

    /** Bitwise complement of an INT32 tensor. */
    public Tensor bitwiseNot() {
        return org.beehive.jitnn.ops.Bitwise.not(this);
    }

    /** this << o for INT32 tensors, with broadcasting. */
    public Tensor shiftLeft(Tensor o) {
        return org.beehive.jitnn.ops.Bitwise.apply(org.beehive.jitnn.ops.Bitwise.Kind.SHIFT_LEFT, this, o);
    }

    /** this >> o (arithmetic: the sign is kept) for INT32 tensors, with broadcasting. */
    public Tensor shiftRight(Tensor o) {
        return org.beehive.jitnn.ops.Bitwise.apply(org.beehive.jitnn.ops.Bitwise.Kind.SHIFT_RIGHT, this, o);
    }

    /**
     * this - floor(this / o) o, with broadcasting: the remainder with the sign of the divisor (torch.remainder). INT32
     * tensors give an INT32 result (0 where o is 0), with no gradient.
     */
    public Tensor remainder(Tensor o) {
        if (dtype() == DType.I32) {
            return org.beehive.jitnn.ops.Bitwise.apply(org.beehive.jitnn.ops.Bitwise.Kind.REMAINDER, this, o);
        }
        return Binary.apply(Binary.Kind.REMAINDER, this, o);
    }

    /**
     * this - trunc(this / o) o, with broadcasting: the remainder with the sign of the dividend (torch.fmod). INT32
     * tensors give an INT32 result (0 where o is 0), with no gradient.
     */
    public Tensor fmod(Tensor o) {
        if (dtype() == DType.I32) {
            return org.beehive.jitnn.ops.Bitwise.apply(org.beehive.jitnn.ops.Bitwise.Kind.FMOD, this, o);
        }
        return Binary.apply(Binary.Kind.FMOD, this, o);
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

    /** Elements idx[0 .. m-1] (INT32 [m]) of {@code axis}, torch.index_select. */
    public Tensor indexSelect(int axis, Tensor idx) {
        return org.beehive.jitnn.ops.Indexing.indexSelect(this, axis, idx);
    }

    /** torch.gather along {@code axis} with an INT32 index of this shape except along the axis. */
    public Tensor gather(int axis, Tensor index) {
        return org.beehive.jitnn.ops.Indexing.gather(this, axis, index);
    }

    /** torch.scatter_add: this with {@code src} added at {@code index} along {@code axis}. */
    public Tensor scatterAdd(int axis, Tensor index, Tensor src) {
        return org.beehive.jitnn.ops.Indexing.scatter(this, axis, index, src, true);
    }

    /** torch.scatter: this with {@code src} written at {@code index} (duplicates: the last wins); no gradient. */
    public Tensor scatter(int axis, Tensor index, Tensor src) {
        return org.beehive.jitnn.ops.Indexing.scatter(this, axis, index, src, false);
    }

    /** Inclusive cumulative sum along {@code axis}. */
    public Tensor cumsum(int axis) {
        return org.beehive.jitnn.ops.Ordered.scan(this, axis, org.beehive.jitnn.kernels.ScanKernels.SUM);
    }

    /** Inclusive cumulative product along {@code axis}; its gradient is exact with zeros. */
    public Tensor cumprod(int axis) {
        return org.beehive.jitnn.ops.Ordered.scan(this, axis, org.beehive.jitnn.kernels.ScanKernels.PROD);
    }

    /** Stable sort along {@code axis}: {values, INT32 original indices}; values carry the gradient. */
    /** INT32 coordinates [count, rank] of the non-zero elements (torch.nonzero); reads the count back to the host. */
    public Tensor nonzero() {
        return org.beehive.jitnn.ops.Compact.nonzero(this);
    }

    /** The elements where {@code mask} (broadcast) is not zero, as a vector (torch.masked_select); reads the count back to the host. */
    public Tensor maskedSelect(Tensor mask) {
        return org.beehive.jitnn.ops.Compact.maskedSelect(this, mask);
    }

    public Tensor[] sort(int axis, boolean descending) {
        return org.beehive.jitnn.ops.Ordered.sort(this, axis, descending);
    }

    /** INT32 indices that sort along {@code axis} (stable). */
    public Tensor argsort(int axis, boolean descending) {
        return sort(axis, descending)[1];
    }

    /** 1 where NaN, else 0 (tested by bit pattern). */
    public Tensor isNaN() {
        return Unary.apply(Unary.Kind.IS_NAN, this, 0, 0);
    }

    /** 1 where +inf or -inf, else 0. */
    public Tensor isInf() {
        return Unary.apply(Unary.Kind.IS_INF, this, 0, 0);
    }

    /** 1 where finite, else 0. */
    public Tensor isFiniteMask() {
        return Unary.apply(Unary.Kind.IS_FINITE, this, 0, 0);
    }

    /** NaN to {@code nan}, +inf to {@code posInf} and -inf to -posInf (torch.nan_to_num with a symmetric infinity). */
    public Tensor nanToNum(float nan, float posInf) {
        return Unary.apply(Unary.Kind.NAN_TO_NUM, this, nan, posInf);
    }

    /** Element-wise maximum propagating NaN (torch.maximum). */
    public Tensor maximumNaN(Tensor o) {
        return Binary.apply(Binary.Kind.MAXIMUM, this, o);
    }

    /** Element-wise minimum propagating NaN (torch.minimum). */
    public Tensor minimumNaN(Tensor o) {
        return Binary.apply(Binary.Kind.MINIMUM, this, o);
    }

    /** Element-wise maximum ignoring a NaN operand (torch.fmax). */
    public Tensor fmax(Tensor o) {
        return Binary.apply(Binary.Kind.FMAX, this, o);
    }

    /** Element-wise minimum ignoring a NaN operand (torch.fmin). */
    public Tensor fmin(Tensor o) {
        return Binary.apply(Binary.Kind.FMIN, this, o);
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

    /** INT32 index of the first minimum along {@code axis}. */
    public Tensor argmin(int axis) {
        return Reduce.argmin(this, axis);
    }

    /** Product along {@code axis}. */
    public Tensor prod(int axis, boolean keepDim) {
        return Reduce.apply(Reduce.Kind.PROD, this, axis, keepDim);
    }

    /** Product of all elements. */
    public Tensor prod() {
        return reshape(numel()).prod(0, false);
    }

    /** The p-norm (p = 1 or 2) along {@code axis}: sum |x| or sqrt(sum x^2). */
    public Tensor norm(int p, int axis, boolean keepDim) {
        return switch (p) {
            case 1 -> abs().sum(axis, keepDim);
            case 2 -> square().sum(axis, keepDim).sqrt();
            default -> throw new IllegalArgumentException("norm: p must be 1 or 2, got " + p);
        };
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
    /**
     * This tensor repeated {@code reps[i]} times along each axis (torch.Tensor.repeat, numpy.tile,
     * ONNX Tile); reps longer than the rank adds leading axes.
     */
    public Tensor tile(int... reps) {
        int r = Math.max(rank(), reps.length);
        int[] d = new int[r];
        int[] rp = new int[r];
        int[] own = shape().dims();
        for (int i = 0; i < r; i++) {
            d[i] = i < r - own.length ? 1 : own[i - (r - own.length)];
            rp[i] = i < r - reps.length ? 1 : reps[i - (r - reps.length)];
        }
        // [d0, d1, ...] -> [1, d0, 1, d1, ...] expanded to [r0, d0, r1, d1, ...] -> [r0 d0, r1 d1, ...]
        int[] split = new int[2 * r];
        int[] big = new int[2 * r];
        int[] out = new int[r];
        for (int i = 0; i < r; i++) {
            split[2 * i] = 1;
            split[2 * i + 1] = d[i];
            big[2 * i] = rp[i];
            big[2 * i + 1] = d[i];
            out[i] = rp[i] * d[i];
        }
        return reshape(split).expand(Shape.of(big)).reshape(out);
    }

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

    /**
     * This tensor converted to {@code dtype}: between FP32, FP16 and BF16 (round to nearest even;
     * differentiable), or between FP32 and INT32 (truncating toward zero; no gradient).
     */
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
