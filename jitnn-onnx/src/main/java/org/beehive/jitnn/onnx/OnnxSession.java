package org.beehive.jitnn.onnx;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import org.beehive.jitnn.Autocast;
import org.beehive.jitnn.DType;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.onnx.OnnxModel.Node;
import org.beehive.jitnn.onnx.OnnxModel.TensorData;

/**
 * Runs an ONNX model with jitNN: each node becomes jitNN operations, so a call records one trace
 * that jitNN compiles into a TornadoVM plan on the first run and replays afterwards (same input
 * shapes). Weights are loaded once into persistent tensors that stay on the device.
 *
 * <pre>{@code
 * OnnxSession s = OnnxSession.load(Path.of("resnet50.onnx"));
 * Tensor logits = s.run(Map.of("x", images)).get("y");
 * }</pre>
 *
 * Floating-point weights of any precision are held in FP32; with {@link #load(Path, boolean)} in
 * half precision the matrix products take FP16 operands (jitNN's autocast) and linear weights are
 * stored in FP16. Convolutions stay FP32. Integer tensors become INT32
 * on the device. Booleans are FP32 0 / 1.
 */
public final class OnnxSession implements AutoCloseable {

    private final OnnxModel model;
    private final Map<String, Value> constants = new HashMap<>();
    private final List<Node> nodes;
    private final boolean half;

    private OnnxSession(OnnxModel model, boolean half) {
        this.model = model;
        this.half = half;
        TreeSet<String> missing = new TreeSet<>();
        for (Node n : model.graph().nodes()) {
            if (!n.opType().equals("Constant") && !OnnxOps.supports(n.opType())) {
                missing.add(n.opType());
            }
            if (!n.domain().isEmpty() && !n.domain().equals("ai.onnx")) {
                missing.add(n.domain() + "." + n.opType());
            }
        }
        if (!missing.isEmpty()) {
            throw new UnsupportedOperationException("ONNX operators not supported by jitNN: " + missing);
        }
        for (TensorData t : model.graph().initializers().values()) {
            constants.put(t.name(), constant(t));
        }
        for (Node n : model.graph().nodes()) {
            if (n.opType().equals("Constant") && n.outputs().size() == 1) {
                constants.put(n.outputs().get(0), constantNode(n));
            }
        }
        java.util.Set<String> outs = new java.util.HashSet<>(outputNames());
        nodes = Boolean.parseBoolean(System.getProperty("jitnn.onnx.rewrite", "true")) ? Rewrites.apply(model.graph().nodes(), constants, outs, half) : model.graph().nodes();
    }

    public static OnnxSession load(Path file) throws IOException {
        return load(file, false);
    }

    /**
     * With {@code halfPrecision}, the matrix products (linear layers, attention) run on FP16
     * operands with FP32 accumulation, and the linear layers' weights are stored in FP16.
     */
    public static OnnxSession load(Path file, boolean halfPrecision) throws IOException {
        return new OnnxSession(OnnxModel.load(file), halfPrecision);
    }

    /** A session over a model built in memory (see {@link OnnxModel#of}). */
    public static OnnxSession of(OnnxModel model, boolean halfPrecision) {
        return new OnnxSession(model, halfPrecision);
    }

    public OnnxModel model() {
        return model;
    }

    public List<String> inputNames() {
        List<String> names = new ArrayList<>();
        model.graph().inputs().forEach(v -> names.add(v.name()));
        return names;
    }

    public List<String> outputNames() {
        List<String> names = new ArrayList<>();
        model.graph().outputs().forEach(v -> names.add(v.name()));
        return names;
    }

    static Value constant(TensorData t) {
        int[] dims = OnnxOps.toInts(t.dims());
        switch (t.dataType()) {
            case OnnxModel.FLOAT, OnnxModel.FLOAT16, OnnxModel.BFLOAT16, OnnxModel.DOUBLE -> {
                float[] f = t.floats();
                Tensor host = Tensor.of(f.length == 0 ? new float[1] : f, dims.length == 0 ? new int[] { 1 } : dims);
                Tensor p = Tensor.persistent(host.reshape(dims));
                return Value.constant(p, f);
            }
            default -> {
                return Value.host(t.longs(), t.dims());
            }
        }
    }

    /**
     * Releases the model's weights: plans compiled for this session can then run another session
     * of the same model (loading it again) without compiling. The session cannot run afterwards.
     */
    @Override
    public void close() {
        for (Value v : constants.values()) {
            if (v.tensor != null && v.tensor.storage().isPersistent()) {
                v.tensor.release();
            }
        }
    }

    /**
     * Runs the graph on {@code inputs} (by graph input name) and returns every graph output, as
     * lazy jitNN tensors (read them with toFloatArray, or keep computing with them).
     */
    public Map<String, Tensor> run(Map<String, Tensor> inputs) {
        Map<String, Value> env = new HashMap<>(constants);
        for (OnnxModel.ValueInfo vi : model.graph().inputs()) {
            Tensor t = inputs.get(vi.name());
            if (t == null) {
                throw new IllegalArgumentException("missing ONNX input '" + vi.name() + "'; the model takes " + inputNames());
            }
            if (vi.elemType() == OnnxModel.INT64 || vi.elemType() == OnnxModel.INT32) {
                if (t.dtype() != DType.I32) {
                    throw new IllegalArgumentException("ONNX input '" + vi.name() + "' is an integer tensor: pass INT32");
                }
                env.put(vi.name(), Value.of(t, OnnxModel.INT64));
            } else {
                env.put(vi.name(), Value.of(t.dtype() == DType.F32 ? t : t.to(DType.F32), OnnxModel.FLOAT));
            }
        }
        Map<String, Tensor> result = new LinkedHashMap<>();
        // intermediates are internal (their buffers are reused, and jitNN's inference fusions apply), as in Grad.inference
        try (Grad.Scope ignored = Grad.disabled(); AutoCloseable internal = org.beehive.jitnn.engine.Storage.internalScope();
                Grad.Scope cast = half ? Autocast.enable(DType.F16) : null) {
            for (Node n : nodes) {
                if (n.opType().equals("Constant")) {
                    continue;
                }
                List<Value> in = new ArrayList<>();
                for (String name : n.inputs()) {
                    if (name.isEmpty()) {
                        in.add(null);
                        continue;
                    }
                    Value v = env.get(name);
                    if (v == null) {
                        throw new IllegalStateException("ONNX node '" + n.name() + "' reads '" + name + "', which nothing produced");
                    }
                    in.add(v);
                }
                List<Value> out;
                try {
                    out = OnnxOps.HANDLERS.get(n.opType()).apply(n, in);
                } catch (RuntimeException e) {
                    throw new IllegalStateException("ONNX " + n.opType() + " node '" + n.name() + "' with inputs " + in + ": " + e.getMessage(), e);
                }
                for (int k = 0; k < n.outputs().size() && k < out.size(); k++) {
                    if (!n.outputs().get(k).isEmpty()) {
                        env.put(n.outputs().get(k), out.get(k));
                    }
                }
            }
            for (OnnxModel.ValueInfo vi : model.graph().outputs()) {
                Value v = env.get(vi.name());
                result.put(vi.name(), v.tensor(v.type != OnnxModel.INT64 && v.type != OnnxModel.INT32));
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        for (Tensor t : result.values()) {
            t.storage().markVisible();
        }
        return result;
    }

    private static Value constantNode(Node n) {
        OnnxModel.Attribute a;
        if ((a = n.attr("value")) != null) {
            return constant(a.t());
        }
        if ((a = n.attr("value_float")) != null) {
            return Value.constant(Tensor.of(new float[] { a.f() }, 1).reshape(), new float[] { a.f() });
        }
        if ((a = n.attr("value_floats")) != null) {
            return Value.constant(Tensor.of(a.floats(), a.floats().length), a.floats());
        }
        if ((a = n.attr("value_int")) != null) {
            return Value.scalar(a.i());
        }
        if ((a = n.attr("value_ints")) != null) {
            return Value.host(a.ints(), new long[] { a.ints().length });
        }
        throw new UnsupportedOperationException("ONNX Constant node '" + n.name() + "' without a value, value_float(s) or value_int(s)");
    }
}
