package org.beehive.jitnn.tf;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.beehive.jitnn.onnx.OnnxModel;
import org.beehive.jitnn.onnx.OnnxModel.Attribute;
import org.beehive.jitnn.onnx.OnnxModel.TensorData;
import org.beehive.jitnn.tf.TfGraph.Node;

/**
 * Rewrites a flat TensorFlow graph as an ONNX graph for jitnn-onnx, which then runs it with its
 * operator handlers, host shape folding and rewrites (as tf2onnx does for ONNX Runtime). NHWC
 * convolutions and pooling are wrapped in transposes to and from NCHW; constant convolution and
 * batch-normalisation weights are rearranged at import. Constant operands that ONNX takes as
 * attributes (permutations, axes) must be constants in the TensorFlow graph.
 */
final class Translator {

    private final Inliner.Result flat;
    private final Map<String, TfGraph.Tensor> consts = new HashMap<>();
    private final Map<String, TensorData> inits = new LinkedHashMap<>();
    private final List<OnnxModel.Node> nodes = new ArrayList<>();
    private final List<OnnxModel.ValueInfo> inputs = new ArrayList<>();
    private final Map<String, Node> byName = new HashMap<>();
    private final Set<String> missing = new TreeSet<>();
    private int fresh;

    Translator(Inliner.Result flat, Map<String, TfGraph.Tensor> variables) {
        this.flat = flat;
        for (Node n : flat.nodes()) {
            byName.put(n.name(), n);
        }
        for (Map.Entry<String, TfGraph.Tensor> v : variables.entrySet()) {
            constant(v.getKey() + ":0", v.getValue());
        }
    }

    // ---------------------------------------------------------------- graph building

    static int onnxType(int tf) {
        return switch (tf) {
            case TfGraph.DT_FLOAT -> OnnxModel.FLOAT;
            case TfGraph.DT_DOUBLE -> OnnxModel.DOUBLE;
            case TfGraph.DT_INT32 -> OnnxModel.INT32;
            case TfGraph.DT_INT64 -> OnnxModel.INT64;
            case TfGraph.DT_UINT8 -> OnnxModel.UINT8;
            case TfGraph.DT_INT8 -> OnnxModel.INT8;
            case TfGraph.DT_BOOL -> OnnxModel.BOOL;
            case TfGraph.DT_HALF -> OnnxModel.FLOAT16;
            case TfGraph.DT_BFLOAT16 -> OnnxModel.BFLOAT16;
            case TfGraph.DT_INT16 -> 5;
            default -> throw new UnsupportedOperationException("TensorFlow data type " + tf);
        };
    }

    private void constant(String name, TfGraph.Tensor t) {
        consts.put(name, t);
        if (t.dtype() != TfGraph.DT_STRING && t.dtype() != TfGraph.DT_RESOURCE) {
            inits.put(name, new TensorData(name, t.dims(), onnxType(t.dtype()), t.raw()));
        }
    }

    private String fresh(String base) {
        return base + "/jitnn_" + (fresh++);
    }

    private String constF(String base, long[] dims, float... values) {
        ByteBuffer b = ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            b.putFloat(v);
        }
        b.rewind();
        String name = fresh(base);
        constant(name, new TfGraph.Tensor(TfGraph.DT_FLOAT, dims, b));
        return name;
    }

    private String constI(String base, long... values) {
        ByteBuffer b = ByteBuffer.allocate(8 * values.length).order(ByteOrder.LITTLE_ENDIAN);
        for (long v : values) {
            b.putLong(v);
        }
        b.rewind();
        String name = fresh(base);
        constant(name, new TfGraph.Tensor(TfGraph.DT_INT64, new long[] { values.length }, b));
        return name;
    }

    private static Attribute ai(String name, long v) {
        return new Attribute(name, 2, 0, v, null, null, null, null, null);
    }

    private static Attribute af(String name, float v) {
        return new Attribute(name, 1, v, 0, null, null, null, null, null);
    }

    private static Attribute as(String name, String v) {
        return new Attribute(name, 3, 0, 0, v, null, null, null, null);
    }

    private static Attribute ais(String name, long... v) {
        return new Attribute(name, 7, 0, 0, null, null, null, v, null);
    }

    private static Attribute at(String name, TensorData t) {
        return new Attribute(name, 4, 0, 0, null, t, null, null, null);
    }

    private String emit(String op, String output, List<String> in, Attribute... attrs) {
        Map<String, Attribute> a = new LinkedHashMap<>();
        for (Attribute x : attrs) {
            a.put(x.name(), x);
        }
        nodes.add(new OnnxModel.Node(output, op, "", in, List.of(output), a));
        return output;
    }

    private void emitMulti(String op, String name, List<String> in, List<String> outs, Attribute... attrs) {
        Map<String, Attribute> a = new LinkedHashMap<>();
        for (Attribute x : attrs) {
            a.put(x.name(), x);
        }
        nodes.add(new OnnxModel.Node(name, op, "", in, outs, a));
    }

    private String tmp(String op, String base, List<String> in, Attribute... attrs) {
        return emit(op, fresh(base), in, attrs);
    }

    /** The constant behind a reference, or null. */
    private TfGraph.Tensor constOf(String ref) {
        return consts.get(ref);
    }

    private long[] constLongs(Node n, String ref, String what) {
        TfGraph.Tensor t = constOf(ref);
        if (t == null) {
            throw new UnsupportedOperationException("TensorFlow " + n.op() + " node " + n.name() + ": the " + what + " must be a constant");
        }
        return t.longs();
    }

    // ---------------------------------------------------------------- translation

    OnnxModel translate(Map<String, String> signatureInputs, Map<String, String> signatureOutputs) {
        // inputs first, so that their names are known as graph inputs
        Map<String, String> placeholderToInput = new HashMap<>();
        for (Map.Entry<String, String> e : signatureInputs.entrySet()) {
            placeholderToInput.put(flat.resolve(e.getValue()), e.getKey());
        }
        for (Node n : flat.nodes()) {
            try {
                node(n, placeholderToInput);
            } catch (UnsupportedOperationException ex) {
                missing.add(ex.getMessage());
            }
        }
        if (!missing.isEmpty()) {
            throw new UnsupportedOperationException("TensorFlow operations not supported by jitNN: " + missing);
        }
        List<OnnxModel.ValueInfo> outputs = new ArrayList<>();
        for (Map.Entry<String, String> e : signatureOutputs.entrySet()) {
            String src = flat.resolve(e.getValue());
            emit("Identity", e.getKey(), List.of(src));
            outputs.add(new OnnxModel.ValueInfo(e.getKey(), OnnxModel.FLOAT, null));
        }
        return OnnxModel.of(new OnnxModel.Graph("tensorflow", prune(outputs), inits, inputs, outputs), Map.of("", 18L));
    }

    /** Only the nodes the outputs need (drops saver, restore and unused branches). */
    private List<OnnxModel.Node> prune(List<OnnxModel.ValueInfo> outputs) {
        Map<String, OnnxModel.Node> producer = new HashMap<>();
        for (OnnxModel.Node n : nodes) {
            for (String o : n.outputs()) {
                producer.put(o, n);
            }
        }
        Set<OnnxModel.Node> keep = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        java.util.ArrayDeque<String> work = new java.util.ArrayDeque<>();
        outputs.forEach(o -> work.add(o.name()));
        while (!work.isEmpty()) {
            OnnxModel.Node n = producer.get(work.pop());
            if (n != null && keep.add(n)) {
                n.inputs().forEach(i -> {
                    if (!i.isEmpty()) {
                        work.add(i);
                    }
                });
            }
        }
        List<OnnxModel.Node> kept = new ArrayList<>();
        for (OnnxModel.Node n : nodes) {
            if (keep.contains(n)) {
                kept.add(n);
            }
        }
        inits.keySet().retainAll(neededInputs(kept));
        return kept;
    }

    private static Set<String> neededInputs(List<OnnxModel.Node> kept) {
        Set<String> s = new java.util.HashSet<>();
        kept.forEach(n -> s.addAll(n.inputs()));
        return s;
    }

    private void node(Node n, Map<String, String> placeholderToInput) {
        String out = n.name() + ":0";
        List<String> in = new ArrayList<>();
        for (String ref : n.data()) {
            in.add(flat.resolve(ref));
        }
        switch (n.op()) {
            case "Const" -> constant(out, n.attr("value").tensor());
            case "Placeholder", "PlaceholderWithDefault" -> {
                if (n.op().equals("PlaceholderWithDefault") && !placeholderToInput.containsKey(out)) {
                    emit("Identity", out, in);
                    return;
                }
                int t = n.type("dtype");
                if (t == TfGraph.DT_RESOURCE) {
                    return;
                }
                String inputName = placeholderToInput.getOrDefault(out, n.name());
                long[] dims = n.attr("shape") == null ? null : n.attr("shape").shape();
                boolean integer = t == TfGraph.DT_INT32 || t == TfGraph.DT_INT64;
                inputs.add(new OnnxModel.ValueInfo(inputName, integer ? OnnxModel.INT64 : OnnxModel.FLOAT, dims));
                if (!inputName.equals(out)) {
                    emit("Identity", out, List.of(inputName));
                }
            }
            case "VarHandleOp", "NoOp", "Assert", "SaveV2", "RestoreV2", "AssignVariableOp", "MergeV2Checkpoints", "ShardedFilename", "StringJoin", "StaticRegexFullMatch",
                    "VarIsInitializedOp", "DisableCopyOnRead" -> {
                // variables are constants once read from the checkpoint; the saver graph is dropped
            }
            case "ReadVariableOp", "Identity", "StopGradient", "Snapshot", "PreventGradient", "CheckNumerics", "EnsureShape" -> {
                String src = n.op().equals("ReadVariableOp") ? TfGraphs.node(in.get(0)) + ":0" : in.get(0);
                TfGraph.Tensor c = constOf(src);
                if (c != null) {
                    // constants stay constants through identities and variable reads
                    constant(out, c);
                } else {
                    emit("Identity", out, List.of(src));
                }
            }
            case "IdentityN" -> {
                for (int i = 0; i < in.size(); i++) {
                    emit("Identity", n.name() + ":" + i, List.of(in.get(i)));
                }
            }
            // ---------------------------------------------------- element-wise
            case "Add", "AddV2" -> emit("Add", out, in);
            case "Sub" -> emit("Sub", out, in);
            case "Mul" -> emit("Mul", out, in);
            case "RealDiv", "Div", "DivNoNan" -> emit("Div", out, in);
            case "FloorDiv" -> emit("Floor", out, List.of(tmp("Div", n.name(), in)));
            case "FloorMod" -> emit("Mod", out, in, ai("fmod", 0));
            case "Maximum" -> emit("Max", out, in);
            case "Minimum" -> emit("Min", out, in);
            case "Pow" -> emit("Pow", out, in);
            case "SquaredDifference" -> {
                String d = tmp("Sub", n.name(), in);
                emit("Mul", out, List.of(d, d));
            }
            case "AddN" -> {
                String acc = in.get(0);
                for (int i = 1; i < in.size(); i++) {
                    acc = i == in.size() - 1 ? emit("Add", out, List.of(acc, in.get(i))) : tmp("Add", n.name(), List.of(acc, in.get(i)));
                }
                if (in.size() == 1) {
                    emit("Identity", out, in);
                }
            }
            case "BiasAdd", "BiasAddV1" -> {
                if (n.s("data_format", "NHWC").equals("NCHW")) {
                    String b = tmp("Unsqueeze", n.name(), List.of(in.get(1), constI(n.name(), 1, 2)));
                    emit("Add", out, List.of(in.get(0), b));
                } else {
                    emit("Add", out, in);
                }
            }
            case "Neg", "Abs", "Exp", "Log", "Sqrt", "Floor", "Ceil", "Round", "Sign", "Sin", "Cos", "Tan", "Tanh", "Sigmoid", "Relu", "Erf", "Softplus", "Softsign", "Elu",
                    "Selu", "Reciprocal" -> emit(n.op(), out, List.of(in.get(0)));
            case "Inv" -> emit("Reciprocal", out, List.of(in.get(0)));
            case "Erfc" -> emit("Sub", out, List.of(constF(n.name(), new long[0], 1), tmp("Erf", n.name(), List.of(in.get(0)))));
            case "Rsqrt" -> emit("Reciprocal", out, List.of(tmp("Sqrt", n.name(), List.of(in.get(0)))));
            case "Square" -> emit("Mul", out, List.of(in.get(0), in.get(0)));
            case "Log1p" -> emit("Log", out, List.of(tmp("Add", n.name(), List.of(in.get(0), constF(n.name(), new long[0], 1)))));
            case "Expm1" -> emit("Sub", out, List.of(tmp("Exp", n.name(), List.of(in.get(0))), constF(n.name(), new long[0], 1)));
            case "Relu6" -> emit("Clip", out, List.of(in.get(0), constF(n.name(), new long[0], 0), constF(n.name(), new long[0], 6)));
            case "LeakyRelu" -> emit("LeakyRelu", out, List.of(in.get(0)), af("alpha", n.f("alpha", 0.2f)));
            case "Softmax" -> emit("Softmax", out, List.of(in.get(0)), ai("axis", -1));
            case "LogSoftmax" -> emit("LogSoftmax", out, List.of(in.get(0)), ai("axis", -1));
            case "Less" -> emit("Less", out, in);
            case "LessEqual" -> emit("LessOrEqual", out, in);
            case "Greater" -> emit("Greater", out, in);
            case "GreaterEqual" -> emit("GreaterOrEqual", out, in);
            case "Equal" -> emit("Equal", out, in.subList(0, 2));
            case "NotEqual" -> emit("Not", out, List.of(tmp("Equal", n.name(), in.subList(0, 2))));
            case "LogicalAnd" -> emit("And", out, in);
            case "LogicalOr" -> emit("Or", out, in);
            case "LogicalNot" -> emit("Not", out, in);
            case "Select", "SelectV2" -> emit("Where", out, in);
            case "Cast" -> {
                int to = n.type("DstT");
                emit("Cast", out, List.of(in.get(0)), ai("to", onnxType(to)));
            }
            // ---------------------------------------------------- products
            case "MatMul" -> {
                String a = n.b("transpose_a", false) ? tmp("Transpose", n.name(), List.of(in.get(0)), ais("perm", 1, 0)) : in.get(0);
                String b = n.b("transpose_b", false) ? tmp("Transpose", n.name(), List.of(in.get(1)), ais("perm", 1, 0)) : in.get(1);
                emit("MatMul", out, List.of(a, b));
            }
            case "BatchMatMul", "BatchMatMulV2", "BatchMatMulV3" -> {
                String a = n.b("adj_x", false) ? tmp("Transpose", n.name(), List.of(in.get(0)), ai("jitnn_last2", 1)) : in.get(0);
                String b = n.b("adj_y", false) ? tmp("Transpose", n.name(), List.of(in.get(1)), ai("jitnn_last2", 1)) : in.get(1);
                emit("MatMul", out, List.of(a, b));
            }
            case "Einsum" -> emit("Einsum", out, in, as("equation", n.s("equation", "")));
            // ---------------------------------------------------- convolution and pooling
            case "Conv2D", "DepthwiseConv2dNative" -> conv(n, in, out);
            case "MaxPool", "AvgPool" -> pool(n, in, out);
            case "FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3" -> batchNorm(n, in, out);
            // ---------------------------------------------------- reductions
            case "Mean", "Sum", "Max", "Min", "Prod" -> {
                String op = switch (n.op()) {
                    case "Mean" -> "ReduceMean";
                    case "Sum" -> "ReduceSum";
                    case "Max" -> "ReduceMax";
                    case "Min" -> "ReduceMin";
                    default -> "ReduceProd";
                };
                long[] axes = constLongs(n, in.get(1), "reduction axes");
                emit(op, out, List.of(in.get(0), constI(n.name(), axes)), ai("keepdims", n.b("keep_dims", false) ? 1 : 0));
            }
            case "ArgMax", "ArgMin" -> {
                long axis = constLongs(n, in.get(1), "axis")[0];
                emit(n.op(), out, List.of(in.get(0)), ai("axis", axis), ai("keepdims", 0));
            }
            case "Cumsum" -> emit("CumSum", out, in, ai("exclusive", n.b("exclusive", false) ? 1 : 0), ai("reverse", n.b("reverse", false) ? 1 : 0));
            case "TopKV2" -> emitMulti("TopK", n.name(), List.of(in.get(0), tmp("Reshape", n.name(), List.of(in.get(1), constI(n.name(), 1)))), List.of(out, n.name() + ":1"),
                    ai("axis", -1));
            // ---------------------------------------------------- shapes
            case "Reshape" -> emit("Reshape", out, in);
            case "Shape" -> emit("Shape", out, List.of(in.get(0)));
            case "Size" -> emit("Size", out, List.of(in.get(0)));
            case "ExpandDims" -> emit("Unsqueeze", out, List.of(in.get(0), constI(n.name(), constLongs(n, in.get(1), "axis"))));
            case "Squeeze" -> {
                long[] dims = n.ints("squeeze_dims");
                emit("Squeeze", out, dims == null || dims.length == 0 ? List.of(in.get(0)) : List.of(in.get(0), constI(n.name(), dims)));
            }
            case "Transpose" -> {
                TfGraph.Tensor perm = constOf(in.get(1));
                // a permutation computed in the graph (Keras Tensordot) is read when the graph runs
                emit("Transpose", out, perm != null ? List.of(in.get(0)) : List.of(in.get(0), in.get(1)), perm != null ? new Attribute[] { ais("perm", perm.longs()) } : new Attribute[0]);
            }
            case "ConcatV2" -> {
                long axis = constLongs(n, in.get(in.size() - 1), "axis")[0];
                emit("Concat", out, in.subList(0, in.size() - 1), ai("axis", axis));
            }
            case "Pack" -> {
                long axis = n.i("axis", 0);
                List<String> parts = new ArrayList<>();
                for (String x : in) {
                    parts.add(tmp("Unsqueeze", n.name(), List.of(x, constI(n.name(), axis))));
                }
                emit("Concat", out, parts, ai("axis", axis));
            }
            case "Unpack" -> {
                long axis = n.i("axis", 0);
                int num = (int) n.i("num", 1);
                List<String> parts = new ArrayList<>();
                for (int i = 0; i < num; i++) {
                    parts.add(fresh(n.name()));
                }
                emitMulti("Split", fresh(n.name()), List.of(in.get(0)), parts, ai("axis", axis));
                for (int i = 0; i < num; i++) {
                    emit("Squeeze", n.name() + ":" + i, List.of(parts.get(i), constI(n.name(), axis)));
                }
            }
            case "Split" -> {
                long axis = constLongs(n, in.get(0), "axis")[0];
                int num = (int) n.i("num_split", 1);
                List<String> outs = new ArrayList<>();
                for (int i = 0; i < num; i++) {
                    outs.add(n.name() + ":" + i);
                }
                emitMulti("Split", fresh(n.name()), List.of(in.get(1)), outs, ai("axis", axis));
            }
            case "SplitV" -> {
                long axis = constLongs(n, in.get(2), "axis")[0];
                long[] sizes = constLongs(n, in.get(1), "split sizes");
                List<String> outs = new ArrayList<>();
                for (int i = 0; i < sizes.length; i++) {
                    outs.add(n.name() + ":" + i);
                }
                emitMulti("Split", fresh(n.name()), List.of(in.get(0), constI(n.name(), sizes)), outs, ai("axis", axis));
            }
            case "Slice" -> {
                long[] begin = constLongs(n, in.get(1), "begin");
                long[] size = constLongs(n, in.get(2), "size");
                long[] end = new long[begin.length];
                long[] axes = new long[begin.length];
                for (int i = 0; i < begin.length; i++) {
                    end[i] = size[i] < 0 ? Long.MAX_VALUE : begin[i] + size[i];
                    axes[i] = i;
                }
                emit("Slice", out, List.of(in.get(0), constI(n.name(), begin), constI(n.name(), end), constI(n.name(), axes)));
            }
            case "StridedSlice" -> stridedSlice(n, in, out);
            case "GatherV2", "Gather", "ResourceGather" -> {
                long axis = n.op().equals("GatherV2") ? constLongs(n, in.get(2), "axis")[0] : 0;
                if (n.i("batch_dims", 0) != 0) {
                    throw new UnsupportedOperationException("TensorFlow " + n.op() + " with batch_dims");
                }
                String data = n.op().equals("ResourceGather") ? TfGraphs.node(in.get(0)) + ":0" : in.get(0);
                emit("Gather", out, List.of(data, in.get(1)), ai("axis", axis));
            }
            case "GatherNd" -> emit("GatherND", out, in);
            case "Tile" -> emit("Tile", out, in);
            case "Fill" -> {
                TfGraph.Tensor v = constOf(in.get(1));
                if (v == null) {
                    emit("Expand", out, List.of(in.get(1), in.get(0)));
                } else {
                    TensorData value = new TensorData("value", new long[] { 1 }, onnxType(v.dtype()), v.raw());
                    emit("ConstantOfShape", out, List.of(in.get(0)), at("value", value));
                }
            }
            case "ZerosLike" -> emit("Mul", out, List.of(in.get(0), constF(n.name(), new long[0], 0)));
            case "OnesLike" -> emit("Add", out, List.of(tmp("Mul", n.name(), List.of(in.get(0), constF(n.name(), new long[0], 0))), constF(n.name(), new long[0], 1)));
            case "Range" -> emit("Range", out, in);
            case "OneHot" -> {
                // where(indices[..., None] == range(depth), on, off)
                long axis = n.i("axis", -1);
                if (axis != -1) {
                    throw new UnsupportedOperationException("TensorFlow OneHot on an axis other than the last");
                }
                String idx = tmp("Unsqueeze", n.name(), List.of(in.get(0), constI(n.name(), -1)));
                String range = tmp("Range", n.name(), List.of(constI(n.name(), 0), in.get(1), constI(n.name(), 1)));
                String eq = tmp("Equal", n.name(), List.of(idx, range));
                emit("Where", out, List.of(eq, in.get(2), in.get(3)));
            }
            case "Pad", "PadV2", "MirrorPad" -> {
                long[] p = constLongs(n, in.get(1), "paddings");
                int r = p.length / 2;
                long[] pads = new long[2 * r];
                for (int i = 0; i < r; i++) {
                    pads[i] = p[2 * i];
                    pads[i + r] = p[2 * i + 1];
                }
                String mode = n.op().equals("MirrorPad") ? (n.s("mode", "REFLECT").equals("REFLECT") ? "reflect" : "edge") : "constant";
                List<String> args = new ArrayList<>(List.of(in.get(0), constI(n.name(), pads)));
                if (n.op().equals("PadV2")) {
                    args.add(in.get(2));
                }
                // NHWC images pad their spatial axes 1 and 2: pad as NCHW
                if (r == 4 && pads[3] == 0 && pads[7] == 0 && (pads[1] != 0 || pads[2] != 0 || pads[5] != 0 || pads[6] != 0)) {
                    String x = toNchw(n, in.get(0));
                    long[] nchw = { 0, 0, pads[1], pads[2], 0, 0, pads[5], pads[6] };
                    args.set(0, x);
                    args.set(1, constI(n.name(), nchw));
                    String y = tmp("Pad", n.name(), args, as("mode", mode));
                    toNhwc(n, y, out);
                } else {
                    emit("Pad", out, args, as("mode", mode));
                }
            }
            case "ResizeBilinear", "ResizeNearestNeighbor" -> {
                long[] size = constLongs(n, in.get(1), "size");
                boolean align = n.b("align_corners", false);
                boolean half = n.b("half_pixel_centers", false);
                boolean bilinear = n.op().equals("ResizeBilinear");
                String ct = align ? "align_corners" : half ? "half_pixel" : "asymmetric";
                String y = tmp("Resize", n.name(), List.of(toNchw(n, in.get(0)), "", "", constI(n.name(), size)), as("mode", bilinear ? "linear" : "nearest"),
                        as("coordinate_transformation_mode", ct), as("nearest_mode", "floor"));
                toNhwc(n, y, out);
            }
            case "DepthToSpace", "SpaceToDepth" -> {
                if (!n.s("data_format", "NHWC").equals("NHWC")) {
                    throw new UnsupportedOperationException("TensorFlow " + n.op() + " in " + n.s("data_format", ""));
                }
                String y = tmp(n.op(), n.name(), List.of(toNchw(n, in.get(0))), ai("blocksize", n.i("block_size", 2)), as("mode", "DCR"));
                toNhwc(n, y, out);
            }
            default -> throw new UnsupportedOperationException(n.op());
        }
    }

    private String toNchw(Node n, String x) {
        return tmp("Transpose", n.name(), List.of(x), ais("perm", 0, 3, 1, 2));
    }

    private void toNhwc(Node n, String y, String out) {
        emit("Transpose", out, List.of(y), ais("perm", 0, 2, 3, 1));
    }

    /** Spatial values of a 4-element NHWC (or NCHW) attribute. */
    private static long[] spatial(long[] v, boolean nchw) {
        return nchw ? new long[] { v[2], v[3] } : new long[] { v[1], v[2] };
    }

    private Attribute[] windowAttrs(Node n, boolean nchw, long[] kernel) {
        List<Attribute> a = new ArrayList<>();
        long[] strides = n.ints("strides");
        a.add(ais("strides", strides == null ? new long[] { 1, 1 } : spatial(strides, nchw)));
        long[] dil = n.ints("dilations");
        if (dil != null && dil.length == 4) {
            a.add(ais("dilations", spatial(dil, nchw)));
        }
        String padding = n.s("padding", "VALID");
        switch (padding) {
            case "SAME" -> a.add(as("auto_pad", "SAME_UPPER"));
            case "VALID" -> a.add(as("auto_pad", "VALID"));
            case "EXPLICIT" -> {
                long[] e = n.ints("explicit_paddings");
                int h = nchw ? 2 : 1;
                int w = nchw ? 3 : 2;
                a.add(ais("pads", e[2 * h], e[2 * w], e[2 * h + 1], e[2 * w + 1]));
            }
            default -> throw new UnsupportedOperationException("TensorFlow padding " + padding);
        }
        if (kernel != null) {
            a.add(ais("kernel_shape", kernel));
        }
        return a.toArray(new Attribute[0]);
    }

    private void conv(Node n, List<String> in, String out) {
        boolean nchw = n.s("data_format", "NHWC").equals("NCHW");
        boolean depthwise = n.op().equals("DepthwiseConv2dNative");
        TfGraph.Tensor w = constOf(in.get(1));
        if (w == null) {
            throw new UnsupportedOperationException("TensorFlow " + n.op() + " with a non-constant filter");
        }
        // HWIO (depthwise: H W C M) to OIHW (depthwise: (C*M) 1 H W)
        long kh = w.dims()[0];
        long kw = w.dims()[1];
        long ci = w.dims()[2];
        long co = w.dims()[3];
        float[] src = w.floats();
        long outCh = depthwise ? ci * co : co;
        long inPerGroup = depthwise ? 1 : ci;
        float[] dst = new float[src.length];
        for (int y = 0; y < kh; y++) {
            for (int x = 0; x < kw; x++) {
                for (int i = 0; i < ci; i++) {
                    for (int o = 0; o < co; o++) {
                        float v = src[(int) (((y * kw + x) * ci + i) * co + o)];
                        long oc = depthwise ? i * co + o : o;
                        long ic = depthwise ? 0 : i;
                        dst[(int) (((oc * inPerGroup + ic) * kh + y) * kw + x)] = v;
                    }
                }
            }
        }
        String weight = constF(n.name(), new long[] { outCh, inPerGroup, kh, kw }, dst);
        List<Attribute> attrs = new ArrayList<>(List.of(windowAttrs(n, nchw, new long[] { kh, kw })));
        if (depthwise) {
            attrs.add(ai("group", ci));
        }
        String x = nchw ? in.get(0) : toNchw(n, in.get(0));
        if (nchw) {
            emit("Conv", out, List.of(x, weight), attrs.toArray(new Attribute[0]));
        } else {
            toNhwc(n, tmp("Conv", n.name(), List.of(x, weight), attrs.toArray(new Attribute[0])), out);
        }
    }

    private void pool(Node n, List<String> in, String out) {
        boolean nchw = n.s("data_format", "NHWC").equals("NCHW");
        long[] k = spatial(n.ints("ksize"), nchw);
        List<Attribute> attrs = new ArrayList<>(List.of(windowAttrs(n, nchw, k)));
        String op = n.op().equals("MaxPool") ? "MaxPool" : "AveragePool";
        if (op.equals("AveragePool")) {
            // TensorFlow averages over the window's in-bounds elements only
            attrs.add(ai("count_include_pad", 0));
        }
        String x = nchw ? in.get(0) : toNchw(n, in.get(0));
        if (nchw) {
            emit(op, out, List.of(x), attrs.toArray(new Attribute[0]));
        } else {
            toNhwc(n, tmp(op, n.name(), List.of(x), attrs.toArray(new Attribute[0])), out);
        }
    }

    private void batchNorm(Node n, List<String> in, String out) {
        if (n.b("is_training", false)) {
            throw new UnsupportedOperationException("TensorFlow " + n.op() + " in training mode");
        }
        boolean nchw = n.s("data_format", "NHWC").equals("NCHW");
        float eps = n.f("epsilon", 1e-3f);
        TfGraph.Tensor gamma = constOf(in.get(1));
        TfGraph.Tensor beta = constOf(in.get(2));
        TfGraph.Tensor mean = constOf(in.get(3));
        TfGraph.Tensor var = constOf(in.get(4));
        if (gamma == null || beta == null || mean == null || var == null) {
            throw new UnsupportedOperationException("TensorFlow " + n.op() + " with non-constant statistics");
        }
        float[] g = gamma.floats();
        float[] b = beta.floats();
        float[] m = mean.floats();
        float[] v = var.floats();
        float[] scale = new float[g.length];
        float[] shift = new float[g.length];
        for (int c = 0; c < g.length; c++) {
            scale[c] = (float) (g[c] / Math.sqrt(v[c] + eps));
            shift[c] = (float) (b[c] - m[c] * (g[c] / Math.sqrt(v[c] + eps)));
        }
        long[] dims = nchw ? new long[] { g.length, 1, 1 } : new long[] { g.length };
        String y = tmp("Mul", n.name(), List.of(in.get(0), constF(n.name(), dims, scale)));
        emit("Add", out, List.of(y, constF(n.name(), dims, shift)));
    }

    /**
     * StridedSlice with constant begin/end/strides as Slice (+ Unsqueeze for new axes, Squeeze for
     * shrunk ones). Dimensions after an ellipsis use negative axes, so the input's rank need not be
     * known.
     */
    private void stridedSlice(Node n, List<String> in, String out) {
        long[] begin = constLongs(n, in.get(1), "begin");
        long[] end = constLongs(n, in.get(2), "end");
        long[] strides = constLongs(n, in.get(3), "strides");
        long beginMask = n.i("begin_mask", 0);
        long endMask = n.i("end_mask", 0);
        long ellipsis = n.i("ellipsis_mask", 0);
        long newAxis = n.i("new_axis_mask", 0);
        long shrink = n.i("shrink_axis_mask", 0);
        int specs = begin.length;
        int ell = Long.numberOfTrailingZeros(ellipsis);
        boolean hasEllipsis = ellipsis != 0;
        // input dimensions consumed by the specs after the ellipsis
        int after = 0;
        if (hasEllipsis) {
            for (int j = ell + 1; j < specs; j++) {
                if ((newAxis >> j & 1) == 0) {
                    after++;
                }
            }
        }
        List<Long> sStarts = new ArrayList<>();
        List<Long> sEnds = new ArrayList<>();
        List<Long> sAxes = new ArrayList<>();
        List<Long> sSteps = new ArrayList<>();
        List<Long> squeeze = new ArrayList<>();
        List<Long> unsqueezeAt = new ArrayList<>();
        int dim = 0;
        int afterSeen = 0;
        for (int j = 0; j < specs; j++) {
            if (hasEllipsis && j == ell) {
                continue;
            }
            boolean pastEllipsis = hasEllipsis && j > ell;
            if ((newAxis >> j & 1) != 0) {
                unsqueezeAt.add((long) j);
                continue;
            }
            long axis = pastEllipsis ? -(after - afterSeen) : dim;
            if (pastEllipsis) {
                afterSeen++;
            } else {
                dim++;
            }
            if ((shrink >> j & 1) != 0) {
                long s = begin[j];
                sStarts.add(s);
                sEnds.add(s == -1 ? Long.MAX_VALUE : s + 1);
                sAxes.add(axis);
                sSteps.add(1L);
                squeeze.add(axis);
                continue;
            }
            long st = strides[j];
            long s = (beginMask >> j & 1) != 0 ? (st > 0 ? 0 : Long.MAX_VALUE) : begin[j];
            long e = (endMask >> j & 1) != 0 ? (st > 0 ? Long.MAX_VALUE : Long.MIN_VALUE) : end[j];
            if ((beginMask >> j & 1) != 0 && (endMask >> j & 1) != 0 && st == 1) {
                continue;
            }
            sStarts.add(s);
            sEnds.add(e);
            sAxes.add(axis);
            sSteps.add(st);
        }
        if (!unsqueezeAt.isEmpty() && hasEllipsis) {
            throw new UnsupportedOperationException("TensorFlow StridedSlice with both an ellipsis and new axes");
        }
        String x = in.get(0);
        if (!sStarts.isEmpty()) {
            x = tmp("Slice", n.name(), List.of(x, constI(n.name(), toArray(sStarts)), constI(n.name(), toArray(sEnds)), constI(n.name(), toArray(sAxes)),
                    constI(n.name(), toArray(sSteps))));
        }
        if (!squeeze.isEmpty()) {
            x = tmp("Squeeze", n.name(), List.of(x, constI(n.name(), toArray(squeeze))));
        }
        if (!unsqueezeAt.isEmpty()) {
            // positions in the output: spec index minus the shrunk specs before it
            long[] at = new long[unsqueezeAt.size()];
            for (int k = 0; k < at.length; k++) {
                long j = unsqueezeAt.get(k);
                long shrunkBefore = Long.bitCount(shrink & ((1L << j) - 1));
                at[k] = j - shrunkBefore;
            }
            x = tmp("Unsqueeze", n.name(), List.of(x, constI(n.name(), at)));
        }
        emit("Identity", out, List.of(x));
    }

    private static long[] toArray(List<Long> l) {
        return l.stream().mapToLong(Long::longValue).toArray();
    }
}
