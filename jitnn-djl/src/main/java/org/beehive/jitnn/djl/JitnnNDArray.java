package org.beehive.jitnn.djl;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.internal.NDFormat;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ops.Binary;

/**
 * A DJL NDArray backed by a lazy jitNN tensor. Operators record jitNN operators and return new
 * arrays; the in-place forms ({@code addi}, {@code set}, ...) swap the array's tensor. Data reaches
 * the host only when it is read ({@code toFloatArray}, {@code toByteBuffer}, {@code getFloat}, ...).
 *
 * <p>The tensor is FP32 for every numeric DJL type; the DJL type is a tag that the results of
 * arithmetic follow (integer division truncates) and that host copies are encoded in. Operators
 * jitNN lacks go to the fallback engine ({@code -Djitnn.djl.fallback}) or throw.
 */
public final class JitnnNDArray extends NDArrayAdapter {

    private Tensor tensor;

    JitnnNDArray(JitnnNDManager manager, Tensor tensor, DataType dataType) {
        super(manager, manager.alternative(), shapeOf(tensor), dataType, UUID.randomUUID().toString());
        this.tensor = tensor;
        manager.attachInternal(uid, this);
    }

    /** The jitNN tensor behind this array (FP32). */
    public Tensor tensor() {
        return tensor;
    }

    // ---------------------------------------------------------------- conversions

    static int[] dims(Shape shape) {
        int[] d = new int[shape.dimension()];
        for (int i = 0; i < d.length; i++) {
            d[i] = Math.toIntExact(shape.get(i));
        }
        return d;
    }

    static Shape shapeOf(Tensor t) {
        long[] d = new long[t.rank()];
        for (int i = 0; i < d.length; i++) {
            d[i] = t.dim(i);
        }
        return new Shape(d);
    }

    JitnnNDManager jm() {
        return (JitnnNDManager) manager;
    }

    JitnnNDArray wrap(Tensor t, DataType type) {
        return jm().wrap(t, type);
    }

    /** A result of the same DJL type as this array (integer types truncate). */
    JitnnNDArray same(Tensor t) {
        return wrap(dataType.isFloating() ? t : t.trunc(), dataType);
    }

    JitnnNDArray floating(Tensor t) {
        return wrap(t, dataType.isFloating() ? dataType : DataType.FLOAT32);
    }

    JitnnNDArray bool(Tensor t) {
        return wrap(t, DataType.BOOLEAN);
    }

    JitnnNDArray index(Tensor t) {
        return wrap(t.dtype() == DType.I32 ? t.to(DType.F32) : t, DataType.INT64);
    }

    /** The jitNN tensor of any NDArray, copying arrays of other engines. */
    Tensor of(NDArray other) {
        return jm().from(other).tensor;
    }

    static Tensor scalar(Number n) {
        return Tensor.scalar(n.floatValue());
    }

    /** INT32 indices on the device. */
    static Tensor ints(Tensor t) {
        return t.dtype() == DType.I32 ? t : t.to(DType.I32);
    }

    private DataType promote(NDArray other) {
        DataType o = other.getDataType();
        if (dataType.isFloating()) {
            return dataType;
        }
        if (o.isFloating()) {
            return o;
        }
        return dataType == DataType.BOOLEAN ? o : dataType;
    }

    private JitnnNDArray arith(NDArray other, Tensor result) {
        DataType type = promote(other);
        return wrap(type.isFloating() ? result : result.trunc(), type);
    }

    /** Makes this array hold {@code result} (the in-place operators); returns this array. */
    JitnnNDArray replace(JitnnNDArray result) {
        tensor = result.tensor;
        dataType = result.dataType;
        shape = result.shape;
        return this;
    }

    /**
     * This array's tensor in persistent storage (it stays on the device and can be updated in
     * place): for optimiser states, running statistics and parameters.
     */
    Tensor makePersistent() {
        if (!tensor.storage().isPersistent()) {
            tensor = tensor.requiresGrad() && tensor.gradFn() == null ? Tensor.parameter(tensor) : Tensor.persistent(tensor);
        }
        return tensor;
    }

    private int axis(int a) {
        return a < 0 ? a + shape.dimension() : a;
    }

    // ---------------------------------------------------------------- resource and data

    @Override
    public NDArrayEx getNDArrayInternal() {
        return new JitnnNDArrayEx(this);
    }

    @Override
    public void intern(NDArray replaced) {
        JitnnNDArray r = jm().from(replaced);
        tensor = r.tensor;
        dataType = r.dataType;
        shape = r.shape;
    }

    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = JitnnNDManager.getSystemManager();
    }

    @Override
    public List<NDArray> getResourceNDArrays() {
        return List.of(this);
    }

    @Override
    public ByteBuffer toByteBuffer(boolean tryDirect) {
        if (dataType == DataType.STRING) {
            throw new UnsupportedOperationException("jitNN arrays hold numbers");
        }
        float[] v = tensor.toFloatArray();
        int n = Math.toIntExact(shape.size());
        ByteBuffer bb = jm().allocateDirect(n * dataType.getNumOfBytes());
        for (int i = 0; i < n; i++) {
            float x = v[i];
            switch (dataType) {
                case FLOAT32 -> bb.putFloat(x);
                case FLOAT64 -> bb.putDouble(x);
                case FLOAT16 -> bb.putShort(Float.floatToFloat16(x));
                case BFLOAT16 -> bb.putShort((short) (Float.floatToRawIntBits(x) >>> 16));
                case INT32, UINT32 -> bb.putInt((int) (long) x);
                case INT64, UINT64 -> bb.putLong((long) x);
                case INT16, UINT16 -> bb.putShort((short) x);
                case INT8, UINT8 -> bb.put((byte) x);
                case BOOLEAN -> bb.put((byte) (x != 0 ? 1 : 0));
                default -> throw new UnsupportedOperationException("jitNN arrays cannot be read as " + dataType);
            }
        }
        bb.rewind();
        return bb;
    }

    @Override
    public float[] toFloatArray() {
        if (dataType == DataType.FLOAT32) {
            return tensor.toFloatArray();
        }
        return super.toFloatArray();
    }

    @Override
    public NDArray toType(DataType type, boolean copy) {
        if (type == dataType && !copy) {
            return this;
        }
        if (type.isInteger() || type.isBoolean()) {
            return wrap(type.isBoolean() ? tensor.ne(Tensor.scalar(0)) : tensor.trunc(), type);
        }
        if (type == DataType.FLOAT16) {
            return wrap(tensor.half().toFloat(), type);
        }
        return wrap(tensor, type);
    }

    @Override
    public NDArray duplicate() {
        return wrap(tensor.storage().isPersistent() ? tensor.mul(1f) : tensor, dataType);
    }

    @Override
    public NDArray toDevice(ai.djl.Device device, boolean copy) {
        return copy ? duplicate() : this;
    }

    @Override
    public String toString() {
        if (isReleased()) {
            return "This array is already closed";
        }
        return NDFormat.format(this, 100, 10, 10, 20);
    }

    // ---------------------------------------------------------------- gradients

    @Override
    public void setRequiresGradient(boolean requiresGrad) {
        if (requiresGrad) {
            if (!tensor.storage().isPersistent()) {
                tensor = Tensor.parameter(tensor);
            } else {
                tensor.requiresGrad(true);
            }
            JitnnEngine.getInstance().track(this);
        } else if (tensor.gradFn() == null) {
            tensor.requiresGrad(false);
        }
    }

    @Override
    public NDArray getGradient() {
        Tensor g = tensor.grad();
        return wrap(g == null ? Tensor.zeros(dims(shape)) : g, DataType.FLOAT32);
    }

    @Override
    public boolean hasGradient() {
        return tensor.requiresGrad();
    }

    @Override
    public NDArray stopGradient() {
        return wrap(tensor.detach(), dataType);
    }

    void clearGradient() {
        if (!isReleased()) {
            tensor.zeroGrad();
        }
    }

    // ---------------------------------------------------------------- comparisons and logic

    @Override
    public NDArray eq(Number n) {
        return bool(tensor.eq(scalar(n)));
    }

    @Override
    public NDArray eq(NDArray other) {
        return bool(tensor.eq(of(other)));
    }

    @Override
    public NDArray neq(Number n) {
        return bool(tensor.ne(scalar(n)));
    }

    @Override
    public NDArray neq(NDArray other) {
        return bool(tensor.ne(of(other)));
    }

    @Override
    public NDArray gt(Number n) {
        return bool(tensor.gt(scalar(n)));
    }

    @Override
    public NDArray gt(NDArray other) {
        return bool(tensor.gt(of(other)));
    }

    @Override
    public NDArray gte(Number n) {
        return bool(tensor.ge(scalar(n)));
    }

    @Override
    public NDArray gte(NDArray other) {
        return bool(tensor.ge(of(other)));
    }

    @Override
    public NDArray lt(Number n) {
        return bool(tensor.lt(scalar(n)));
    }

    @Override
    public NDArray lt(NDArray other) {
        return bool(tensor.lt(of(other)));
    }

    @Override
    public NDArray lte(Number n) {
        return bool(tensor.le(scalar(n)));
    }

    @Override
    public NDArray lte(NDArray other) {
        return bool(tensor.le(of(other)));
    }

    @Override
    public NDArray logicalAnd(NDArray other) {
        return bool(tensor.logicalAnd(of(other)));
    }

    @Override
    public NDArray logicalOr(NDArray other) {
        return bool(tensor.logicalOr(of(other)));
    }

    @Override
    public NDArray logicalXor(NDArray other) {
        return bool(tensor.logicalXor(of(other)));
    }

    @Override
    public NDArray logicalNot() {
        return bool(tensor.logicalNot());
    }

    @Override
    public NDArray isNaN() {
        return bool(tensor.isNaN());
    }

    @Override
    public NDArray isInfinite() {
        return bool(tensor.isInf());
    }

    @Override
    public boolean contentEquals(NDArray other) {
        if (other == null || isReleased() || other.isReleased() || !shape.equals(other.getShape())) {
            return false;
        }
        return Arrays.equals(toFloatArrayAny(this), toFloatArrayAny(other));
    }

    private static float[] toFloatArrayAny(NDArray a) {
        return a instanceof JitnnNDArray j ? j.tensor.toFloatArray() : a.toType(DataType.FLOAT32, false).toFloatArray();
    }

    // ---------------------------------------------------------------- arithmetic

    @Override
    public NDArray add(Number n) {
        return same(tensor.add(n.floatValue()));
    }

    @Override
    public NDArray add(NDArray other) {
        return arith(other, tensor.add(of(other)));
    }

    @Override
    public NDArray sub(Number n) {
        return same(tensor.add(-n.floatValue()));
    }

    @Override
    public NDArray sub(NDArray other) {
        return arith(other, tensor.sub(of(other)));
    }

    @Override
    public NDArray mul(Number n) {
        return same(tensor.mul(n.floatValue()));
    }

    @Override
    public NDArray mul(NDArray other) {
        return arith(other, tensor.mul(of(other)));
    }

    @Override
    public NDArray div(Number n) {
        return same(tensor.div(scalar(n)));
    }

    @Override
    public NDArray div(NDArray other) {
        return arith(other, tensor.div(of(other)));
    }

    @Override
    public NDArray mod(Number n) {
        return same(tensor.remainder(scalar(n)));
    }

    @Override
    public NDArray mod(NDArray other) {
        return arith(other, tensor.remainder(of(other)));
    }

    @Override
    public NDArray pow(Number n) {
        return same(tensor.pow(n.floatValue()));
    }

    @Override
    public NDArray pow(NDArray other) {
        return arith(other, Binary.apply(Binary.Kind.POW, tensor, of(other)));
    }

    @Override
    public NDArray maximum(Number n) {
        return same(tensor.maximum(scalar(n)));
    }

    @Override
    public NDArray maximum(NDArray other) {
        return arith(other, tensor.maximum(of(other)));
    }

    @Override
    public NDArray minimum(Number n) {
        return same(tensor.minimum(scalar(n)));
    }

    @Override
    public NDArray minimum(NDArray other) {
        return arith(other, tensor.minimum(of(other)));
    }

    @Override
    public NDArray atan2(NDArray other) {
        return floating(tensor.atan2(of(other)));
    }

    @Override
    public NDArray addi(Number n) {
        return replace((JitnnNDArray) add(n));
    }

    @Override
    public NDArray addi(NDArray other) {
        return replace((JitnnNDArray) add(other));
    }

    @Override
    public NDArray subi(Number n) {
        return replace((JitnnNDArray) sub(n));
    }

    @Override
    public NDArray subi(NDArray other) {
        return replace((JitnnNDArray) sub(other));
    }

    @Override
    public NDArray muli(Number n) {
        return replace((JitnnNDArray) mul(n));
    }

    @Override
    public NDArray muli(NDArray other) {
        return replace((JitnnNDArray) mul(other));
    }

    @Override
    public NDArray divi(Number n) {
        return replace((JitnnNDArray) div(n));
    }

    @Override
    public NDArray divi(NDArray other) {
        return replace((JitnnNDArray) div(other));
    }

    @Override
    public NDArray modi(Number n) {
        return replace((JitnnNDArray) mod(n));
    }

    @Override
    public NDArray modi(NDArray other) {
        return replace((JitnnNDArray) mod(other));
    }

    @Override
    public NDArray powi(Number n) {
        return replace((JitnnNDArray) pow(n));
    }

    @Override
    public NDArray powi(NDArray other) {
        return replace((JitnnNDArray) pow(other));
    }

    @Override
    public NDArray signi() {
        return replace((JitnnNDArray) sign());
    }

    @Override
    public NDArray negi() {
        return replace((JitnnNDArray) neg());
    }

    // ---------------------------------------------------------------- unary

    @Override
    public NDArray sign() {
        return same(tensor.sign());
    }

    @Override
    public NDArray neg() {
        return same(tensor.neg());
    }

    @Override
    public NDArray abs() {
        return same(tensor.abs());
    }

    @Override
    public NDArray square() {
        return same(tensor.square());
    }

    @Override
    public NDArray sqrt() {
        return floating(tensor.sqrt());
    }

    @Override
    public NDArray cbrt() {
        return floating(tensor.sign().mul(tensor.abs().pow(1f / 3)));
    }

    @Override
    public NDArray floor() {
        return same(tensor.floor());
    }

    @Override
    public NDArray ceil() {
        return same(tensor.ceil());
    }

    @Override
    public NDArray round() {
        return same(tensor.round());
    }

    @Override
    public NDArray trunc() {
        return same(tensor.trunc());
    }

    @Override
    public NDArray exp() {
        return floating(tensor.exp());
    }

    @Override
    public NDArray log() {
        return floating(tensor.log());
    }

    @Override
    public NDArray log10() {
        return floating(tensor.log().mul((float) (1 / Math.log(10))));
    }

    @Override
    public NDArray log2() {
        return floating(tensor.log().mul((float) (1 / Math.log(2))));
    }

    @Override
    public NDArray sin() {
        return floating(tensor.sin());
    }

    @Override
    public NDArray cos() {
        return floating(tensor.cos());
    }

    @Override
    public NDArray tan() {
        return floating(tensor.tan());
    }

    @Override
    public NDArray asin() {
        return floating(tensor.asin());
    }

    @Override
    public NDArray acos() {
        return floating(tensor.acos());
    }

    @Override
    public NDArray atan() {
        return floating(tensor.atan());
    }

    @Override
    public NDArray sinh() {
        return floating(tensor.sinh());
    }

    @Override
    public NDArray cosh() {
        return floating(tensor.cosh());
    }

    @Override
    public NDArray tanh() {
        return floating(tensor.tanh());
    }

    @Override
    public NDArray asinh() {
        return floating(tensor.asinh());
    }

    @Override
    public NDArray acosh() {
        return floating(tensor.acosh());
    }

    @Override
    public NDArray atanh() {
        return floating(tensor.atanh());
    }

    @Override
    public NDArray toDegrees() {
        return floating(tensor.mul((float) (180 / Math.PI)));
    }

    @Override
    public NDArray toRadians() {
        return floating(tensor.mul((float) (Math.PI / 180)));
    }

    @Override
    public NDArray erf() {
        return floating(tensor.erf());
    }

    @Override
    public NDArray erfinv() {
        return floating(erfinv(tensor));
    }

    /** Inverse error function (Giles' single-precision approximation), from jitNN operators. */
    static Tensor erfinv(Tensor x) {
        Tensor w = x.mul(x).neg().add(1).log().neg();
        Tensor small = w.add(-2.5f);
        Tensor p1 = poly(small, 2.81022636e-08f, 3.43273939e-07f, -3.5233877e-06f, -4.39150654e-06f, 0.00021858087f, -0.00125372503f, -0.00417768164f, 0.246640727f, 1.50140941f);
        Tensor large = w.sqrt().add(-3);
        Tensor p2 = poly(large, -0.000200214257f, 0.000100950558f, 0.00134934322f, -0.00367342844f, 0.00573950773f, -0.0076224613f, 0.00943887047f, 1.00167406f, 2.83297682f);
        return F.where(w.lt(Tensor.scalar(5)), p1, p2).mul(x);
    }

    private static Tensor poly(Tensor w, float... c) {
        Tensor p = Tensor.full(c[0], w.shape().dims());
        for (int i = 1; i < c.length; i++) {
            p = p.mul(w).add(c[i]);
        }
        return p;
    }

    @Override
    public NDArray clip(Number min, Number max) {
        return same(tensor.clamp(min.floatValue(), max.floatValue()));
    }

    // ---------------------------------------------------------------- reductions

    private int[] allAxes() {
        int[] a = new int[shape.dimension()];
        for (int i = 0; i < a.length; i++) {
            a[i] = i;
        }
        return a;
    }

    private interface Reducer {
        Tensor apply(Tensor t, int axis, boolean keepDim);
    }

    private Tensor reduce(Reducer r, int[] axes, boolean keepDims) {
        int[] sorted = axes.clone();
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = axis(sorted[i]);
        }
        Arrays.sort(sorted);
        Tensor t = tensor;
        for (int i = sorted.length - 1; i >= 0; i--) {
            t = r.apply(t, sorted[i], keepDims);
        }
        return t;
    }

    @Override
    public NDArray max() {
        return same(tensor.reshape(-1).max(0, false));
    }

    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return same(reduce(Tensor::max, axes, keepDims));
    }

    @Override
    public NDArray min() {
        return same(tensor.reshape(-1).min(0, false));
    }

    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return same(reduce(Tensor::min, axes, keepDims));
    }

    @Override
    public NDArray sum() {
        return wrap(tensor.sum(), sumType());
    }

    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return wrap(reduce(Tensor::sum, axes, keepDims), sumType());
    }

    private DataType sumType() {
        return dataType == DataType.BOOLEAN ? DataType.INT64 : dataType;
    }

    @Override
    public NDArray prod() {
        return wrap(tensor.prod(), sumType());
    }

    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return wrap(reduce(Tensor::prod, axes, keepDims), sumType());
    }

    @Override
    public NDArray mean() {
        return floating(tensor.mean());
    }

    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return floating(reduce(Tensor::mean, axes, keepDims));
    }

    @Override
    public NDArray cumSum() {
        return wrap(tensor.reshape(-1).cumsum(0), sumType());
    }

    @Override
    public NDArray cumSum(int axis) {
        return wrap(tensor.cumsum(axis(axis)), sumType());
    }

    @Override
    public NDArray cumProd(int axis) {
        return wrap(tensor.cumprod(axis(axis)), sumType());
    }

    @Override
    public NDArray cumProd(int axis, DataType type) {
        return wrap(tensor.cumprod(axis(axis)), type);
    }

    @Override
    public NDArray argMax() {
        return index(tensor.reshape(-1).argmax(0));
    }

    @Override
    public NDArray argMax(int axis) {
        return index(tensor.argmax(axis(axis)));
    }

    @Override
    public NDArray argMin() {
        return index(tensor.reshape(-1).argmin(0));
    }

    @Override
    public NDArray argMin(int axis) {
        return index(tensor.argmin(axis(axis)));
    }

    @Override
    public NDArray norm(boolean keepDims) {
        Tensor n = tensor.reshape(-1).norm(2, 0, false);
        return floating(keepDims ? n.reshape(ones(shape.dimension())) : n);
    }

    @Override
    public NDArray norm(int ord, int[] axes, boolean keepDims) {
        if (axes.length != 1) {
            return super.norm(ord, axes, keepDims);
        }
        return floating(tensor.norm(ord, axis(axes[0]), keepDims));
    }

    @Override
    public NDArray normalize(double p, long dim, double eps) {
        Tensor n = tensor.norm((int) p, axis((int) dim), true).maximum(Tensor.scalar((float) eps));
        return floating(tensor.div(n));
    }

    private static int[] ones(int n) {
        int[] d = new int[n];
        Arrays.fill(d, 1);
        return d;
    }

    // ---------------------------------------------------------------- shapes

    @Override
    public NDArray reshape(Shape newShape) {
        return wrap(tensor.reshape(dims(newShape)), dataType);
    }

    @Override
    public NDArray flatten() {
        return wrap(tensor.reshape(-1), dataType);
    }

    @Override
    public NDArray flatten(int startDim, int endDim) {
        int s = axis(startDim);
        int e = axis(endDim);
        int[] d = new int[shape.dimension() - (e - s)];
        int k = 0;
        for (int i = 0; i < s; i++) {
            d[k++] = (int) shape.get(i);
        }
        int merged = 1;
        for (int i = s; i <= e; i++) {
            merged *= (int) shape.get(i);
        }
        d[k++] = merged;
        for (int i = e + 1; i < shape.dimension(); i++) {
            d[k++] = (int) shape.get(i);
        }
        return wrap(tensor.reshape(d), dataType);
    }

    @Override
    public NDArray expandDims(int axis) {
        return wrap(tensor.unsqueeze(axis < 0 ? axis + shape.dimension() + 1 : axis), dataType);
    }

    @Override
    public NDArray squeeze(int[] axes) {
        int[] sorted = axes.clone();
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = axis(sorted[i]);
        }
        Arrays.sort(sorted);
        Tensor t = tensor;
        for (int i = sorted.length - 1; i >= 0; i--) {
            t = t.squeeze(sorted[i]);
        }
        return wrap(t, dataType);
    }

    @Override
    public NDArray squeeze() {
        int[] d = Arrays.stream(dims(shape)).filter(x -> x != 1).toArray();
        return wrap(tensor.reshape(d), dataType);
    }

    @Override
    public NDArray transpose() {
        int r = shape.dimension();
        int[] order = new int[r];
        for (int i = 0; i < r; i++) {
            order[i] = r - 1 - i;
        }
        return transpose(order);
    }

    @Override
    public NDArray transpose(int... axes) {
        int[] order = axes.clone();
        for (int i = 0; i < order.length; i++) {
            order[i] = axis(order[i]);
        }
        return wrap(tensor.permute(order), dataType);
    }

    @Override
    public NDArray broadcast(Shape target) {
        return wrap(tensor.expand(org.beehive.jitnn.Shape.of(dims(target))), dataType);
    }

    @Override
    public NDList split(long sections, int axis) {
        int ax = axis(axis);
        int size = (int) shape.get(ax);
        if (size % sections != 0) {
            throw new IllegalArgumentException("axis " + axis + " of " + shape + " does not split into " + sections + " equal sections");
        }
        int[] sizes = new int[(int) sections];
        Arrays.fill(sizes, (int) (size / sections));
        return list(tensor.split(ax, sizes));
    }

    @Override
    public NDList split(long[] indices, int axis) {
        int ax = axis(axis);
        int size = (int) shape.get(ax);
        long[] bounds = indices.length > 0 && indices[0] == 0 ? indices : prepend0(indices);
        int[] sizes = new int[bounds.length];
        for (int i = 0; i < bounds.length; i++) {
            long end = i + 1 < bounds.length ? bounds[i + 1] : size;
            sizes[i] = (int) Math.max(0, Math.min(end, size) - Math.min(bounds[i], size));
        }
        return list(tensor.split(ax, sizes));
    }

    private static long[] prepend0(long[] a) {
        long[] b = new long[a.length + 1];
        System.arraycopy(a, 0, b, 1, a.length);
        return b;
    }

    private NDList list(Tensor[] parts) {
        NDList out = new NDList(parts.length);
        for (Tensor p : parts) {
            out.add(wrap(p, dataType));
        }
        return out;
    }

    @Override
    public NDArray tile(long repeats) {
        int[] r = ones(shape.dimension());
        Arrays.fill(r, (int) repeats);
        return wrap(tensor.tile(r), dataType);
    }

    @Override
    public NDArray tile(int axis, long repeats) {
        int[] r = ones(shape.dimension());
        r[axis(axis)] = (int) repeats;
        return wrap(tensor.tile(r), dataType);
    }

    @Override
    public NDArray tile(long[] repeats) {
        return wrap(tensor.tile(Arrays.stream(repeats).mapToInt(Math::toIntExact).toArray()), dataType);
    }

    @Override
    public NDArray tile(Shape desired) {
        long[] r = new long[desired.dimension()];
        int offset = desired.dimension() - shape.dimension();
        for (int i = 0; i < r.length; i++) {
            long have = i < offset ? 1 : shape.get(i - offset);
            r[i] = desired.get(i) / have;
        }
        return tile(r);
    }

    @Override
    public NDArray repeat(int axis, long repeats) {
        int ax = axis(axis);
        int[] d = dims(shape);
        Tensor t = tensor.unsqueeze(ax + 1);
        int[] target = new int[d.length + 1];
        for (int i = 0, k = 0; i < target.length; i++) {
            target[i] = i == ax + 1 ? (int) repeats : d[k++];
        }
        t = t.expand(org.beehive.jitnn.Shape.of(target));
        d[ax] *= (int) repeats;
        return wrap(t.reshape(d), dataType);
    }

    @Override
    public NDArray repeat(long repeats) {
        NDArray r = this;
        for (int i = 0; i < shape.dimension(); i++) {
            r = r.repeat(i, repeats);
        }
        return r;
    }

    @Override
    public NDArray repeat(long[] repeats) {
        NDArray r = this;
        for (int i = 0; i < repeats.length; i++) {
            r = r.repeat(i, repeats[i]);
        }
        return r;
    }

    @Override
    public NDArray flip(int... axes) {
        int[] a = axes.clone();
        for (int i = 0; i < a.length; i++) {
            a[i] = axis(a[i]);
        }
        return wrap(F.flip(tensor, a), dataType);
    }

    // ---------------------------------------------------------------- linear algebra

    @Override
    public NDArray matMul(NDArray other) {
        Tensor b = of(other);
        Tensor a = tensor;
        boolean vecA = a.rank() == 1;
        boolean vecB = b.rank() == 1;
        if (vecA) {
            a = a.unsqueeze(0);
        }
        if (vecB) {
            b = b.unsqueeze(1);
        }
        Tensor y = a.matmul(b);
        if (vecA) {
            y = y.squeeze(-2);
        }
        if (vecB) {
            y = y.squeeze(-1);
        }
        return arith(other, y);
    }

    @Override
    public NDArray dot(NDArray other) {
        if (shape.dimension() == 1 && other.getShape().dimension() == 1) {
            return arith(other, tensor.mul(of(other)).sum());
        }
        if (shape.isScalar() || other.getShape().isScalar()) {
            return mul(other);
        }
        return matMul(other);
    }

    @Override
    public NDArray batchDot(NDArray other) {
        return matMul(other);
    }

    @Override
    public NDArray batchMatMul(NDArray other) {
        return matMul(other);
    }

    // ---------------------------------------------------------------- softmax, sorting

    private Tensor lastAxis(int axis, java.util.function.UnaryOperator<Tensor> f) {
        int ax = axis(axis);
        int last = shape.dimension() - 1;
        if (ax == last) {
            return f.apply(tensor);
        }
        return f.apply(tensor.transpose(ax, last)).transpose(ax, last);
    }

    @Override
    public NDArray softmax(int axis) {
        return floating(lastAxis(axis, Tensor::softmax));
    }

    @Override
    public NDArray logSoftmax(int axis) {
        return floating(lastAxis(axis, Tensor::logSoftmax));
    }

    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return index(tensor.argsort(axis(axis), !ascending));
    }

    @Override
    public NDArray sort(int axis) {
        return wrap(tensor.sort(axis(axis), false)[0], dataType);
    }

    @Override
    public NDArray sort() {
        return sort(-1);
    }

    @Override
    public NDList topK(int k, int axis, boolean largest, boolean sorted) {
        int ax = axis(axis);
        int last = shape.dimension() - 1;
        Tensor t = ax == last ? tensor : tensor.transpose(ax, last);
        Tensor[] r = (largest ? t : t.neg()).topk(k);
        Tensor values = largest ? r[0] : r[0].neg();
        Tensor indices = r[1];
        if (ax != last) {
            values = values.transpose(ax, last);
            indices = indices.transpose(ax, last);
        }
        return new NDList(wrap(values, dataType), index(indices));
    }

    // ---------------------------------------------------------------- indexing

    private JitnnIndexer indexer() {
        return new JitnnIndexer(jm());
    }

    @Override
    public NDArray get(NDIndex index) {
        return indexer().get(this, index);
    }

    @Override
    public void set(NDIndex index, NDArray value) {
        indexer().set(this, index, value);
    }

    @Override
    public void set(NDIndex index, Number value) {
        indexer().set(this, index, value);
    }

    @Override
    public void set(NDIndex index, java.util.function.Function<NDArray, NDArray> function) {
        set(index, function.apply(get(index)));
    }

    @Override
    public void set(NDArray index, Number value) {
        Tensor mask = of(index);
        replace(wrap(F.where(mask.ne(Tensor.scalar(0)), Tensor.full(value.floatValue(), dims(shape)), tensor), dataType));
    }

    @Override
    public void setScalar(NDIndex index, Number value) {
        indexer().setScalar(this, index, value);
    }

    @Override
    public NDArray gather(NDArray index, int axis) {
        return wrap(tensor.gather(axis(axis), ints(of(index))), dataType);
    }

    @Override
    public NDArray gatherNd(NDArray index) {
        // DJL indexes with the first axis of index (MXNet's layout); jitNN with the last (ONNX's)
        Tensor idx = ints(of(index));
        int r = idx.rank();
        int[] order = new int[r];
        for (int i = 0; i < r - 1; i++) {
            order[i] = i + 1;
        }
        order[r - 1] = 0;
        return wrap(F.gatherND(tensor, r == 1 ? idx.reshape(1, (int) idx.dim(0)) : idx.permute(order)), dataType);
    }

    @Override
    public NDArray take(NDManager m, NDArray index) {
        Tensor idx = ints(of(index));
        Tensor flat = tensor.reshape(-1).indexSelect(0, idx.reshape(-1));
        return jm().wrap(flat.reshape(idx.shape().dims()), dataType);
    }

    @Override
    public NDArray put(NDArray index, NDArray value) {
        Tensor idx = ints(of(index)).reshape(-1);
        Tensor v = of(value).reshape(-1);
        return wrap(tensor.reshape(-1).scatter(0, idx, v).reshape(dims(shape)), dataType);
    }

    @Override
    public NDArray scatter(NDArray index, NDArray value, int axis) {
        return wrap(tensor.scatter(axis(axis), ints(of(index)), of(value)), dataType);
    }

    @Override
    public NDArray oneHot(int depth, float onValue, float offValue, DataType type) {
        Tensor oh = F.oneHot(ints(tensor), depth);
        if (onValue != 1 || offValue != 0) {
            oh = oh.mul(onValue - offValue).add(offValue);
        }
        return wrap(oh, type);
    }

    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        int t = (int) shape.get(1);
        int[] md = ones(shape.dimension());
        md[1] = t;
        Tensor positions = Tensor.arange(0, t, 1).reshape(md);
        int[] ld = ones(shape.dimension());
        ld[0] = (int) shape.get(0);
        Tensor keep = positions.lt(of(sequenceLength).reshape(ld));
        Tensor full = Tensor.full(value, dims(shape));
        return wrap(F.where(keep.expand(tensor.shape()), tensor, full), dataType);
    }

    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        if (axis != 0 || !index.getShape().equals(shape)) {
            return super.booleanMask(index, axis);
        }
        return wrap(tensor.maskedSelect(of(index)), dataType);
    }

    @Override
    public NDArray nonzero() {
        return index(tensor.nonzero());
    }

    @Override
    public NDArray pad(Shape padding, double value) {
        // DJL (as PyTorch) lists the pads of the last axis first: (left, right, top, bottom, ...)
        int r = shape.dimension();
        int pairs = padding.dimension() / 2;
        if (r - pairs < 2) {
            return super.pad(padding, value);
        }
        int[] before = new int[r - 2];
        int[] after = new int[r - 2];
        for (int i = 0; i < pairs; i++) {
            before[r - 3 - i] = (int) padding.get(2 * i);
            after[r - 3 - i] = (int) padding.get(2 * i + 1);
        }
        return wrap(F.pad(tensor, before, after, F.PadMode.CONSTANT, (float) value), dataType);
    }
}
