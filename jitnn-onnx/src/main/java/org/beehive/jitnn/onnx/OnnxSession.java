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
 * half precision activations stay FP16 between operators ({@code Autocast.enableHalf}), linear
 * weights are stored in FP16, and outputs are handed out in FP32. {@code -Djitnn.onnx.halfActivations=false}
 * keeps FP32 activations with FP16 products only (jitNN's autocast). Integer tensors become INT32
 * on the device. Booleans are FP32 0 / 1.
 */
public final class OnnxSession implements AutoCloseable {

    private final OnnxModel model;
    private final Map<String, Value> constants = new HashMap<>();
    private final List<Node> nodes;
    private final boolean half;
    /**
     * Half precision keeps activations in FP16 ({@code -Djitnn.onnx.halfActivations}, default true):
     * BERT-base on an A10 ran 1.5-2.0x faster than with FP16 products alone.
     */
    private static final boolean HALF_ACTIVATIONS = !"false".equals(System.getProperty("jitnn.onnx.halfActivations"));

    private OnnxSession(OnnxModel model, boolean half) {
        this.model = model;
        this.half = half;
        for (TensorData t : model.graph().initializers().values()) {
            constants.put(t.name(), constant(t));
        }
        for (Node n : model.graph().nodes()) {
            if (n.opType().equals("Constant") && n.outputs().size() == 1) {
                constants.put(n.outputs().get(0), constantNode(n));
            }
        }
        // quantised operators become float ones around (de)quantisation; constant weights are dequantised once here
        java.util.Set<String> graphOutputs = new java.util.HashSet<>();
        for (OnnxModel.ValueInfo vi : model.graph().outputs()) {
            graphOutputs.add(vi.name());
        }
        List<Node> lowered = OnnxQuant.lower(model.graph().nodes(), constants, graphOutputs);
        TreeSet<String> missing = new TreeSet<>();
        for (Node n : lowered) {
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
        java.util.Set<String> outs = new java.util.HashSet<>(outputNames());
        nodes = Boolean.parseBoolean(System.getProperty("jitnn.onnx.rewrite", "true")) ? Rewrites.apply(lowered, constants, outs, half) : lowered;
    }

    public static OnnxSession load(Path file) throws IOException {
        return load(file, false);
    }

    /**
     * With {@code halfPrecision}, activations stay FP16 between operators (products accumulate in
     * FP32) and the linear layers' weights are stored in FP16; outputs are FP32.
     */
    public static OnnxSession load(Path file, boolean halfPrecision) throws IOException {
        return of(OnnxModel.load(file), halfPrecision);
    }

    /** A session over a model built in memory (see {@link OnnxModel#of}). */
    /** Buckets for every session ({@code -Djitnn.onnx.buckets=1,2,4,8,...}, see {@link #withBuckets}), or none. */
    private static final int[] DEFAULT_BUCKETS = parseBuckets(System.getProperty("jitnn.onnx.buckets", ""));

    private static int[] parseBuckets(String s) {
        if (s.isBlank()) {
            return null;
        }
        return java.util.Arrays.stream(s.split(",")).map(String::trim).filter(x -> !x.isEmpty()).mapToInt(Integer::parseInt).toArray();
    }

    public static OnnxSession of(OnnxModel model, boolean halfPrecision) {
        OnnxSession session = new OnnxSession(model, halfPrecision);
        return DEFAULT_BUCKETS != null ? session.withBuckets(DEFAULT_BUCKETS) : session;
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
            case OnnxModel.INT8, OnnxModel.UINT8, OnnxModel.INT16, OnnxModel.UINT16 -> {
                return Value.host(t.longs(), t.dims(), t.dataType());
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
        if (buckets == null) {
            return runExact(inputs);
        }
        // pad the symbolic dimensions that may be padded to their bucket, run, and cut the outputs back
        Map<String, Integer> actual = new HashMap<>();
        Map<String, Integer> padded = new HashMap<>();
        for (OnnxModel.ValueInfo vi : model.graph().inputs()) {
            Tensor t = inputs.get(vi.name());
            if (t == null || vi.dimNames() == null) {
                continue;
            }
            for (int a = 0; a < vi.dimNames().length && a < t.rank(); a++) {
                String d = vi.dimNames()[a];
                if (d != null && paddable.contains(d)) {
                    actual.put(d, t.dim(a));
                    padded.put(d, bucket(t.dim(a)));
                }
            }
        }
        Map<String, Tensor> in = new HashMap<>(inputs);
        for (OnnxModel.ValueInfo vi : model.graph().inputs()) {
            Tensor t = in.get(vi.name());
            if (t == null || vi.dimNames() == null) {
                continue;
            }
            for (int a = 0; a < vi.dimNames().length && a < t.rank(); a++) {
                String d = vi.dimNames()[a];
                Integer b = d == null ? null : padded.get(d);
                if (b != null && b > t.dim(a)) {
                    int[] dims = t.shape().dims();
                    dims[a] = b - t.dim(a);
                    t = t.dtype() == DType.I32 ? padIntsWithZeros(t, a, b) : Tensor.cat(a, t, Tensor.zeros(dims).to(t.dtype()));
                }
            }
            in.put(vi.name(), t);
        }
        // where each padded dimension sits in the inputs, for outputs whose dimensions carry no name
        Map<Integer, String> axisOf = new HashMap<>();
        for (OnnxModel.ValueInfo vi : model.graph().inputs()) {
            for (int a = 0; vi.dimNames() != null && a < vi.dimNames().length; a++) {
                String d = vi.dimNames()[a];
                if (d != null && padded.containsKey(d)) {
                    axisOf.putIfAbsent(a, d);
                }
            }
        }
        Map<String, Tensor> out = runExact(in);
        // the bucket's plan runs as it is (one plan per bucket); the outputs are cut back on the host
        // as they are read out, so no operator depends on the request's own sizes
        org.beehive.jitnn.engine.Engine.sync(out.values().toArray(new Tensor[0]));
        for (OnnxModel.ValueInfo vo : model.graph().outputs()) {
            Tensor t = out.get(vo.name());
            if (t == null) {
                continue;
            }
            int[] keep = t.shape().dims();
            boolean cut = false;
            for (int a = 0; a < t.rank(); a++) {
                String d = vo.dimNames() != null && a < vo.dimNames().length ? vo.dimNames()[a] : null;
                if (d == null || !padded.containsKey(d)) {
                    String guess = axisOf.get(a);
                    d = guess != null && t.dim(a) == padded.get(guess) ? guess : d;
                }
                Integer n = d == null ? null : actual.get(d);
                if (n != null && t.dim(a) > n) {
                    keep[a] = n;
                    cut = true;
                }
            }
            if (cut) {
                out.put(vo.name(), cutOnHost(t, keep));
            }
        }
        return out;
    }

    private int[] buckets;
    /** Symbolic dimensions padding cannot change the results of: see {@link #withBuckets}. */
    private final java.util.Set<String> paddable = new java.util.HashSet<>();

    /**
     * Pads the model's symbolic dimensions to the smallest of {@code sizes} that holds them (a size
     * beyond the largest is run as it is), so a server compiles one plan per bucket instead of one
     * per request shape. Only dimensions padding cannot change the results of are padded:
     * <ul>
     * <li>a dimension that is the first axis of every input it appears in (the batch: rows are
     * computed independently), and</li>
     * <li>a dimension of an input whose name contains {@code mask} (a sequence: padded positions are
     * masked out, as zeros in the mask).</li>
     * </ul>
     * Inputs are padded with zeros and outputs cut back to the request's sizes. Returns this session.
     */
    public OnnxSession withBuckets(int... sizes) {
        int[] b = sizes.clone();
        java.util.Arrays.sort(b);
        buckets = b;
        paddable.clear();
        Map<String, Boolean> firstAxisOnly = new HashMap<>();
        for (OnnxModel.ValueInfo vi : model.graph().inputs()) {
            if (vi.dimNames() == null) {
                continue;
            }
            for (int a = 0; a < vi.dimNames().length; a++) {
                String d = vi.dimNames()[a];
                if (d == null) {
                    continue;
                }
                firstAxisOnly.merge(d, a == 0, Boolean::logicalAnd);
                if (vi.name().toLowerCase(java.util.Locale.ROOT).contains("mask")) {
                    paddable.add(d);
                }
            }
        }
        firstAxisOnly.forEach((d, first) -> {
            if (first) {
                paddable.add(d);
            }
        });
        // one plan per combination of buckets: keep them all in the plan cache
        long combos = 1;
        for (int i = 0; i < paddable.size(); i++) {
            combos = Math.min(combos * b.length, 1024);
        }
        org.beehive.jitnn.engine.Engine.reservePlans((int) combos);
        return this;
    }

    /** The symbolic dimensions {@link #withBuckets} pads (for tests and logging). */
    public java.util.Set<String> paddedDimensions() {
        return java.util.Collections.unmodifiableSet(paddable);
    }

    /** The leading {@code keep[a]} entries of every axis a of t, as a host tensor (INT32 stays INT32, others become FP32). */
    private static Tensor cutOnHost(Tensor t, int[] keep) {
        int[] dims = t.shape().dims();
        int count = 1;
        for (int k : keep) {
            count *= k;
        }
        boolean ints = t.dtype() == DType.I32;
        int[] iv = ints ? t.toIntArray() : null;
        float[] fv = ints ? null : t.toFloatArray();
        int[] io = ints ? new int[count] : null;
        float[] fo = ints ? null : new float[count];
        int[] idx = new int[dims.length];
        for (int o = 0; o < count; o++) {
            int src = 0;
            for (int a = 0; a < dims.length; a++) {
                src = src * dims[a] + idx[a];
            }
            if (ints) {
                io[o] = iv[src];
            } else {
                fo[o] = fv[src];
            }
            for (int a = dims.length - 1; a >= 0; a--) {
                if (++idx[a] < keep[a]) {
                    break;
                }
                idx[a] = 0;
            }
        }
        return ints ? Tensor.of(io, keep) : Tensor.of(fo, keep);
    }

    /** t (INT32: token ids, masks; host data) with axis {@code axis} padded with zeros to {@code size}. */
    private static Tensor padIntsWithZeros(Tensor t, int axis, int size) {
        int[] v = t.toIntArray();
        int[] dims = t.shape().dims();
        int outer = 1;
        for (int i = 0; i < axis; i++) {
            outer *= dims[i];
        }
        int inner = 1;
        for (int i = axis + 1; i < dims.length; i++) {
            inner *= dims[i];
        }
        int[] out = new int[outer * size * inner];
        for (int o = 0; o < outer; o++) {
            System.arraycopy(v, o * dims[axis] * inner, out, o * size * inner, dims[axis] * inner);
        }
        dims[axis] = size;
        return Tensor.of(out, dims);
    }

    private int bucket(int n) {
        for (int b : buckets) {
            if (b >= n) {
                return b;
            }
        }
        return n;
    }

    private Map<String, Tensor> runExact(Map<String, Tensor> inputs) {
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
                Grad.Scope cast = !half ? null : HALF_ACTIVATIONS ? Autocast.enableHalf() : Autocast.enable(DType.F16)) {
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
                Tensor t = v.tensor(v.type != OnnxModel.INT64 && v.type != OnnxModel.INT32);
                boolean quantised = v.type == OnnxModel.INT8 || v.type == OnnxModel.UINT8 || v.type == OnnxModel.INT16 || v.type == OnnxModel.UINT16 || v.type == OnnxModel.INT32;
                // quantised values are carried as FP32 integers; integer outputs are handed out as INT32
                if (t.dtype() == DType.F16) {
                    t = t.to(DType.F32);
                }
                result.put(vi.name(), quantised && t.dtype() == DType.F32 ? t.to(DType.I32) : t);
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
