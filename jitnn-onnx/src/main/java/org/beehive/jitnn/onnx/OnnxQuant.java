package org.beehive.jitnn.onnx;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.onnx.OnnxModel.Attribute;
import org.beehive.jitnn.onnx.OnnxModel.Node;
import org.beehive.jitnn.onnx.OnnxOps.Handler;

/**
 * Quantised ONNX models: the QDQ format (QuantizeLinear / DequantizeLinear around float operators,
 * what ONNX Runtime's static quantisation writes), dynamic quantisation (DynamicQuantizeLinear,
 * MatMulInteger, ConvInteger) and the QLinearMatMul / QLinearConv operators.
 *
 * <p>At load, the integer operators are rewritten into DequantizeLinear, the float operator and
 * QuantizeLinear, and every DequantizeLinear of a constant (the weights, the biases) is folded into
 * an FP32 weight. What runs is therefore the float model with the quantisation of the activations
 * applied exactly (scale, round half to even, zero point, saturation); the integer products are
 * computed in FP32, which holds them exactly up to 2^24 per sum, and come out as FP32 (an INT32
 * graph output is converted). Weights are stored dequantised.
 */
final class OnnxQuant {

    private OnnxQuant() {
    }

    private static final int ATTR_INT = 2;

    // ---------------------------------------------------------------- load-time rewriting

    /** Lowers the integer operators and folds the dequantisation of constants into FP32 weights. */
    static List<Node> lower(List<Node> nodes, Map<String, Value> constants) {
        List<Node> out = new ArrayList<>();
        int[] fresh = { 0 };
        for (Node n : nodes) {
            switch (n.opType()) {
                case "QLinearMatMul" -> {
                    String a = dq(out, n, n.input(0), n.input(1), n.input(2), 1, fresh);
                    String b = dq(out, n, n.input(3), n.input(4), n.input(5), lastAxis(constants, n.input(3)), fresh);
                    String y = name(n, "mm", fresh);
                    out.add(node(n, "MatMul", List.of(a, b), List.of(y), Map.of()));
                    out.add(node(n, "QuantizeLinear", List.of(y, n.input(6), n.input(7)), n.outputs(), Map.of()));
                }
                case "QLinearConv" -> {
                    String x = dq(out, n, n.input(0), n.input(1), n.input(2), 1, fresh);
                    String w = dq(out, n, n.input(3), n.input(4), n.input(5), 0, fresh);
                    List<String> conv = new ArrayList<>(List.of(x, w));
                    if (!n.input(8).isEmpty()) {
                        // the INT32 bias has scale x_scale * w_scale and no zero point
                        String scale = name(n, "bias_scale", fresh);
                        constants.put(scale, product(constants, n, n.input(1), n.input(4)));
                        conv.add(dq(out, n, n.input(8), scale, "", 0, fresh));
                    }
                    String y = name(n, "conv", fresh);
                    out.add(node(n, "Conv", conv, List.of(y), n.attributes()));
                    out.add(node(n, "QuantizeLinear", List.of(y, n.input(6), n.input(7)), n.outputs(), Map.of()));
                }
                case "MatMulInteger" -> {
                    String one = one(constants);
                    String a = n.input(2).isEmpty() ? n.input(0) : dq(out, n, n.input(0), one, n.input(2), 0, fresh);
                    String b = n.input(3).isEmpty() ? n.input(1) : dq(out, n, n.input(1), one, n.input(3), lastAxis(constants, n.input(1)), fresh);
                    // a plain MatMul, so a constant (folded) weight becomes a linear layer like any other
                    out.add(node(n, "MatMul", List.of(a, b), n.outputs(), Map.of()));
                }
                case "ConvInteger" -> {
                    String one = one(constants);
                    String x = n.input(2).isEmpty() ? n.input(0) : dq(out, n, n.input(0), one, n.input(2), 1, fresh);
                    String w = n.input(3).isEmpty() ? n.input(1) : dq(out, n, n.input(1), one, n.input(3), 0, fresh);
                    out.add(node(n, "Conv", List.of(x, w), n.outputs(), n.attributes()));
                }
                default -> out.add(n);
            }
        }
        // fold DequantizeLinear of constants (in order, so a folded value can feed the next)
        List<Node> kept = new ArrayList<>();
        for (Node n : out) {
            if (n.opType().equals("DequantizeLinear") && allConstant(constants, n)) {
                List<Value> in = new ArrayList<>();
                for (String s : n.inputs()) {
                    in.add(s.isEmpty() ? null : constants.get(s));
                }
                Tensor t = dequantize(n, in).tensor;
                float[] v = t.toFloatArray();
                constants.put(n.outputs().get(0), Value.constant(Tensor.persistent(t), v));
            } else {
                kept.add(n);
            }
        }
        return kept;
    }

    private static String dq(List<Node> out, Node n, String x, String scale, String zp, int axis, int[] fresh) {
        String y = name(n, "dq", fresh);
        List<String> in = zp.isEmpty() ? List.of(x, scale) : List.of(x, scale, zp);
        out.add(node(n, "DequantizeLinear", in, List.of(y), Map.of("axis", new Attribute("axis", ATTR_INT, 0, axis, null, null, null, null, null))));
        return y;
    }

    private static Node node(Node from, String op, List<String> in, List<String> out, Map<String, Attribute> attrs) {
        return new Node(from.name() + "/" + op, op, "", in, out, attrs);
    }

    private static String name(Node n, String what, int[] fresh) {
        return (n.name().isEmpty() ? n.outputs().get(0) : n.name()) + "#" + what + fresh[0]++;
    }

    private static boolean allConstant(Map<String, Value> constants, Node n) {
        for (String s : n.inputs()) {
            if (!s.isEmpty() && !constants.containsKey(s)) {
                return false;
            }
        }
        return true;
    }

    /** For a per-column zero point or scale of a 2-D constant matrix: its last axis. */
    private static int lastAxis(Map<String, Value> constants, String name) {
        Value v = constants.get(name);
        return v == null ? 1 : Math.max(v.rank() - 1, 0);
    }

    private static String one(Map<String, Value> constants) {
        constants.computeIfAbsent("#one", k -> Value.constant(Tensor.persistent(Tensor.of(new float[] { 1 }, 1).reshape()), new float[] { 1 }));
        return "#one";
    }

    private static Value product(Map<String, Value> constants, Node n, String a, String b) {
        Value va = constants.get(a);
        Value vb = constants.get(b);
        if (va == null || vb == null) {
            throw new UnsupportedOperationException(n.name() + ": QLinearConv with a bias needs constant scales");
        }
        float[] x = floats(va);
        float[] w = floats(vb);
        float[] s = new float[w.length];
        for (int i = 0; i < s.length; i++) {
            s[i] = x[x.length == 1 ? 0 : i] * w[i];
        }
        return Value.constant(Tensor.persistent(s.length == 1 ? Tensor.of(s, 1).reshape() : Tensor.of(s, s.length)), s);
    }

    private static float[] floats(Value v) {
        float[] f = v.hostFloats();
        return f != null ? f : v.tensor.toFloatArray();
    }

    // ---------------------------------------------------------------- operators

    static void register(Map<String, Handler> h) {
        h.put("DequantizeLinear", (n, in) -> List.of(dequantize(n, in)));
        h.put("QuantizeLinear", (n, in) -> {
            int type = (int) n.attrInt("output_dtype", OnnxOps.present(in, 2) ? in.get(2).type : OnnxModel.UINT8);
            if (type == OnnxModel.INT64) {
                // a zero point created by the load-time rewriting carries no element type of its own
                type = OnnxModel.UINT8;
            }
            noBlocks(n);
            float[] range = range(n, type);
            Tensor x = OnnxOps.t(in, 0);
            Tensor q = x.div(perAxis(n, in.get(1), x.rank())).round();
            if (OnnxOps.present(in, 2)) {
                q = q.add(perAxis(n, in.get(2), x.rank()));
            }
            return List.of(Value.of(q.clamp(range[0], range[1]), type));
        });
        h.put("DynamicQuantizeLinear", (n, in) -> {
            Tensor x = OnnxOps.t(in, 0);
            Tensor flat = x.reshape(x.numel());
            Tensor zero = Tensor.scalar(0.0f);
            Tensor lo = flat.min(0, false).minimum(zero);
            Tensor hi = flat.max(0, false).maximum(zero);
            Tensor scale = hi.sub(lo).div(Tensor.scalar(255.0f));
            Tensor zp = zero.sub(lo.div(scale)).round().clamp(0, 255);
            Tensor y = x.div(scale).round().add(zp).clamp(0, 255);
            return List.of(Value.of(y, OnnxModel.UINT8), Value.of(scale, OnnxModel.FLOAT), Value.of(zp, OnnxModel.UINT8));
        });
    }

    static Value dequantize(Node n, List<Value> in) {
        noBlocks(n);
        Value x = in.get(0);
        int rank = x.rank();
        if (x.isHost()) {
            // constants (weights): computed on the host
            float[] v = x.hostFloats();
            float[] s = floats(in.get(1));
            float[] z = OnnxOps.present(in, 2) ? floats(in.get(2)) : new float[] { 0 };
            int ax = s.length == 1 && z.length == 1 ? 0 : OnnxOps.axis(n.attrInt("axis", 1), rank);
            int[] d = x.intDims();
            int inner = 1;
            for (int i = ax + 1; i < rank; i++) {
                inner *= d[i];
            }
            int channels = rank == 0 ? 1 : d[ax];
            float[] out = new float[v.length];
            for (int i = 0; i < v.length; i++) {
                int c = (i / inner) % channels;
                out[i] = (v[i] - z[z.length == 1 ? 0 : c]) * s[s.length == 1 ? 0 : c];
            }
            int[] dims = d.length == 0 ? new int[] { 1 } : d;
            return Value.of(Tensor.of(out, dims).reshape(d), OnnxModel.FLOAT);
        }
        Tensor t = OnnxOps.t(in, 0);
        if (OnnxOps.present(in, 2)) {
            t = t.sub(perAxis(n, in.get(2), rank));
        }
        return Value.of(t.mul(perAxis(n, in.get(1), rank)), OnnxModel.FLOAT);
    }

    /** A scale or zero point: a scalar, or a 1-D tensor along the node's axis, shaped to broadcast. */
    private static Tensor perAxis(Node n, Value v, int rank) {
        Tensor t = v.tensor(true);
        if (t.numel() == 1) {
            return t.reshape();
        }
        if (t.rank() != 1) {
            throw new UnsupportedOperationException(n.name() + ": " + n.opType() + " with a scale or zero point of rank " + t.rank());
        }
        int ax = OnnxOps.axis(n.attrInt("axis", 1), rank);
        int[] dims = new int[rank];
        java.util.Arrays.fill(dims, 1);
        dims[ax] = t.numel();
        return t.reshape(dims);
    }

    private static void noBlocks(Node n) {
        if (n.attrInt("block_size", 0) != 0) {
            throw new UnsupportedOperationException(n.name() + ": blocked " + n.opType());
        }
    }

    private static final Map<Integer, float[]> RANGES = new HashMap<>(Map.of(
            OnnxModel.UINT8, new float[] { 0, 255 },
            OnnxModel.INT8, new float[] { -128, 127 },
            OnnxModel.UINT16, new float[] { 0, 65535 },
            OnnxModel.INT16, new float[] { -32768, 32767 }));

    private static float[] range(Node n, int type) {
        float[] r = RANGES.get(type);
        if (r == null) {
            throw new UnsupportedOperationException(n.name() + ": QuantizeLinear to ONNX type " + type + " (UINT8, INT8, UINT16 and INT16 are supported)");
        }
        return r;
    }
}
