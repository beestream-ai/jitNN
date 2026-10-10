package org.beehive.jitnn;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.Task4;
import uk.ac.manchester.tornado.api.common.TornadoFunctions.Task5;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * An operator defined outside jitNN: a kernel written in plain Java with TornadoVM's
 * {@code KernelContext}, an optional host reference and an optional gradient. Once built, it is
 * recorded, fused into the step's compiled plan, cached and differentiated like any built-in
 * operator.
 *
 * <p>Element-wise operators on FP32 tensors take one line:
 *
 * <pre>{@code
 * public static void swish(KernelContext kc, FloatArray x, FloatArray y, int n) {
 *     int i = kc.globalIdx;
 *     if (i < n) {
 *         float v = x.get(i);
 *         y.set(i, v / (1.0f + TornadoMath.exp(-v)));
 *     }
 * }
 *
 * // d/dx x s(x) = y + s(x) (1 - y)
 * static final CustomOp SWISH = CustomOp.unary("swish", MyKernels::swish, v -> v / (1 + Math.exp(-v)))
 *         .withGradient((x, y, gy) -> gy.mul(y.add(x.sigmoid().mul(y.neg().add(1)))));
 *
 * Tensor y = SWISH.apply(x);
 * }</pre>
 *
 * <p>Anything else goes through {@link #builder(String)}: the output shapes, then the tasks the
 * operator adds to the plan, each with an explicit launch grid. Kernels take the
 * {@code KernelContext} first and bound-check their thread index; they never use {@code @Parallel}.
 *
 * <p>Names are global: compiled plans are cached by the operators' names and attributes, so two
 * operators cannot share a name.
 */
public final class CustomOp {

    /** Adds the operator's device tasks for {@code node} to the plan being built. */
    @FunctionalInterface
    public interface Launcher {
        void emit(Emitter e, Node node);
    }

    /** Computes {@code node} on the host: inputs are readable, outputs are allocated. */
    @FunctionalInterface
    public interface Reference {
        void run(Node node);
    }

    /** Gradients with respect to each input (null where none is needed), given the gradient of the first output. */
    @FunctionalInterface
    public interface Backward {
        Tensor[] apply(Node node, Tensor gradOut);
    }

    /** The shape of an output, from the inputs and the integer attributes of the call. */
    @FunctionalInterface
    public interface OutputShape {
        Shape of(Tensor[] inputs, int[] iattr);
    }

    /** The gradient of an element-wise unary operator y = f(x): dL/dx from x, y and dL/dy. */
    @FunctionalInterface
    public interface UnaryGradient {
        Tensor apply(Tensor x, Tensor y, Tensor gradY);
    }

    /** The gradients of an element-wise binary operator y = f(a, b): [dL/da, dL/db] from a, b, y and dL/dy. */
    @FunctionalInterface
    public interface BinaryGradient {
        Tensor[] apply(Tensor a, Tensor b, Tensor y, Tensor gradY);
    }

    private static final Map<String, CustomOp> NAMES = new ConcurrentHashMap<>();

    private final String name;
    private final List<OutputShape> shapes;
    private final List<DType> dtypes;
    private final DType[] inputTypes;
    private final Launcher launcher;
    private final Reference reference;
    private final Backward backward;
    private final Impl impl = new Impl();

    private CustomOp(String name, List<OutputShape> shapes, List<DType> dtypes, DType[] inputTypes, Launcher launcher, Reference reference, Backward backward) {
        this.name = name;
        this.shapes = List.copyOf(shapes);
        this.dtypes = List.copyOf(dtypes);
        this.inputTypes = inputTypes;
        this.launcher = launcher;
        this.reference = reference;
        this.backward = backward;
    }

    /** Starts the definition of an operator named {@code name}. */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * An element-wise operator on one FP32 tensor. {@code kernel} is called as
     * {@code kernel(kc, x, y, n)} on a 1-D grid of at least n threads; {@code reference} computes
     * one value on the host.
     */
    public static CustomOp unary(String name, Task4<KernelContext, FloatArray, FloatArray, Integer> kernel, DoubleUnaryOperator reference) {
        return builder(name).inputs(DType.F32).output((in, a) -> in[0].shape(), DType.F32).launch((e, n) -> {
            FloatArray x = e.f32(n.in(0));
            FloatArray y = e.f32(n.out(0));
            int count = n.out(0).numel();
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, kernel, e.context(), x, y, count));
        }).reference(reference == null ? null : n -> {
            FloatArray x = (FloatArray) n.in(0).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                y.set(i, (float) reference.applyAsDouble(x.get(i)));
            }
        }).build();
    }

    /**
     * An element-wise operator on two FP32 tensors of the same shape. {@code kernel} is called as
     * {@code kernel(kc, a, b, y, n)} on a 1-D grid of at least n threads.
     */
    public static CustomOp binary(String name, Task5<KernelContext, FloatArray, FloatArray, FloatArray, Integer> kernel, DoubleBinaryOperator reference) {
        return builder(name).inputs(DType.F32, DType.F32).output((in, a) -> {
            if (!in[0].shape().equals(in[1].shape())) {
                throw new IllegalArgumentException(name + ": operands must have the same shape, got " + in[0].shape() + " and " + in[1].shape());
            }
            return in[0].shape();
        }, DType.F32).launch((e, n) -> {
            FloatArray a = e.f32(n.in(0));
            FloatArray b = e.f32(n.in(1));
            FloatArray y = e.f32(n.out(0));
            int count = n.out(0).numel();
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, kernel, e.context(), a, b, y, count));
        }).reference(reference == null ? null : n -> {
            FloatArray a = (FloatArray) n.in(0).storage().host();
            FloatArray b = (FloatArray) n.in(1).storage().host();
            FloatArray y = (FloatArray) n.out(0).storage().host();
            for (int i = 0; i < n.out(0).numel(); i++) {
                y.set(i, (float) reference.applyAsDouble(a.get(i), b.get(i)));
            }
        }).build();
    }

    /** This element-wise unary operator with a gradient. */
    public CustomOp withGradient(UnaryGradient gradient) {
        if (inputTypes.length != 1) {
            throw new IllegalStateException(name + " takes " + inputTypes.length + " inputs, not one");
        }
        return replace((n, g) -> new Tensor[] { gradient.apply(n.in(0), n.out(0), g) });
    }

    /** This element-wise binary operator with a gradient. */
    public CustomOp withGradient(BinaryGradient gradient) {
        if (inputTypes.length != 2) {
            throw new IllegalStateException(name + " takes " + inputTypes.length + " inputs, not two");
        }
        return replace((n, g) -> gradient.apply(n.in(0), n.in(1), n.out(0), g));
    }

    private CustomOp replace(Backward b) {
        CustomOp op = new CustomOp(name, shapes, dtypes, inputTypes, launcher, reference, b);
        NAMES.replace(name, this, op);
        return op;
    }

    public String name() {
        return name;
    }

    /** Applies the operator; returns its first output. */
    public Tensor apply(Tensor... inputs) {
        return applyAll(null, null, inputs)[0];
    }

    /** Applies the operator with integer and float attributes (part of the plan's cache key); returns its first output. */
    public Tensor apply(int[] iattr, float[] fattr, Tensor... inputs) {
        return applyAll(iattr, fattr, inputs)[0];
    }

    /** Applies the operator; returns all its outputs. */
    public Tensor[] applyAll(int[] iattr, float[] fattr, Tensor... inputs) {
        if (inputs.length != inputTypes.length) {
            throw new IllegalArgumentException(name + " takes " + inputTypes.length + " inputs, got " + inputs.length);
        }
        for (int i = 0; i < inputs.length; i++) {
            if (inputTypes[i] != null && inputs[i].dtype() != inputTypes[i]) {
                throw new IllegalArgumentException(name + ": input " + i + " must be " + inputTypes[i] + ", got " + inputs[i]);
            }
        }
        int[] ia = iattr == null ? new int[0] : iattr.clone();
        float[] fa = fattr == null ? new float[0] : fattr.clone();
        Tensor[] outputs = new Tensor[shapes.size()];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = Tensor.result(shapes.get(i).of(inputs, ia), dtypes.get(i));
        }
        Tensor.record(impl, inputs.clone(), outputs, ia, fa, true);
        return outputs;
    }

    @Override
    public String toString() {
        return "CustomOp[" + name + "]";
    }

    /** The operator as the engine sees it. */
    private final class Impl implements Op {

        @Override
        public String name() {
            return "custom:" + name;
        }

        @Override
        public void emit(Emitter e, Node node) {
            launcher.emit(e, node);
        }

        @Override
        public void reference(Node node) {
            if (reference == null) {
                throw new UnsupportedOperationException("custom operator " + name + " has no host reference: run it on a TornadoVM device");
            }
            reference.run(node);
        }

        @Override
        public Tensor[] backward(Node node, Tensor gradOut) {
            if (backward == null) {
                throw new UnsupportedOperationException("custom operator " + name + " has no gradient");
            }
            return backward.apply(node, gradOut);
        }
    }

    /** Defines a custom operator step by step; {@link #build()} registers its name. */
    public static final class Builder {

        private final String name;
        private final List<OutputShape> shapes = new ArrayList<>();
        private final List<DType> dtypes = new ArrayList<>();
        private DType[] inputTypes;
        private Launcher launcher;
        private Reference reference;
        private Backward backward;

        private Builder(String name) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("a custom operator needs a name");
            }
            this.name = name;
        }

        /** The inputs' element types, in order (null accepts any type). */
        public Builder inputs(DType... types) {
            this.inputTypes = types.clone();
            return this;
        }

        /** Adds an output of the given shape and element type. */
        public Builder output(OutputShape shape, DType dtype) {
            shapes.add(shape);
            dtypes.add(dtype);
            return this;
        }

        /** The device tasks: called once per plan, with the arrays bound to the node's tensors. */
        public Builder launch(Launcher launcher) {
            this.launcher = launcher;
            return this;
        }

        /** The host implementation, used by the reference backend and by tests (optional). */
        public Builder reference(Reference reference) {
            this.reference = reference;
            return this;
        }

        /** The gradient, written with other operators (optional). */
        public Builder backward(Backward backward) {
            this.backward = backward;
            return this;
        }

        public CustomOp build() {
            if (inputTypes == null) {
                throw new IllegalStateException(name + ": declare the inputs");
            }
            if (shapes.isEmpty()) {
                throw new IllegalStateException(name + ": declare at least one output");
            }
            if (launcher == null) {
                throw new IllegalStateException(name + ": declare the device tasks with launch(...)");
            }
            CustomOp op = new CustomOp(name, shapes, dtypes, inputTypes, launcher, reference, backward);
            if (NAMES.putIfAbsent(name, op) != null) {
                throw new IllegalArgumentException("a custom operator named " + name + " already exists: compiled plans are cached by operator name");
            }
            return op;
        }
    }
}
