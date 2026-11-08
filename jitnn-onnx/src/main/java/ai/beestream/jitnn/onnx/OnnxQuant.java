/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.onnx;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.JitnnConfig;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.onnx.OnnxModel.Attribute;
import ai.beestream.jitnn.onnx.OnnxModel.Node;
import ai.beestream.jitnn.onnx.OnnxOps.Handler;

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
    private static final int ATTR_INTS = 7;

    // ---------------------------------------------------------------- load-time rewriting

    /**
     * Lowers the integer operators and folds the dequantisation of constants into FP32 weights;
     * {@code outputs} (the graph's outputs) are kept.
     */
    static List<Node> lower(List<Node> nodes, Map<String, Value> constants, java.util.Set<String> outputs) {
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
                    if (INT8_KERNELS && int8Weight(constants, n.input(1)) && zero(constants, n.input(3))) {
                        // INT8 product: a holds the integers already (scale 1, zero point a_zero_point)
                        out.add(int8MatMul(n, n.input(0), one(constants), n.input(2), true, n.input(1), one(constants), constants, n.outputs(), fresh));
                        continue;
                    }
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
        if (INT8_KERNELS) {
            out = int8Products(out, constants, outputs, fresh);
            out = int8Epilogues(out, constants, outputs, fresh);
        }
        out = fakeQuantize(out, constants);
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

    /**
     * Products in INT8 ({@code -Djitnn.onnx.int8=false} keeps them in FP32 on dequantised values):
     * the INT8 tensor-core kernels on CUDA, an INT32 loop elsewhere.
     */
    private static final boolean INT8_KERNELS = JitnnConfig.current().get(JitnnConfig.ONNX_INT8);

    /**
     * MatMul(DequantizeLinear(a, s, z), DequantizeLinear(w, sw)) with a constant INT8 w [K, N] (zero
     * point 0, per-tensor or per-column scale) and a per-tensor s and z: an Int8MatMul. When a is
     * QuantizeLinear(x, s, z), the product quantises x itself and the two nodes go when nothing else
     * reads them; otherwise a holds the integers.
     */
    private static List<Node> int8Products(List<Node> nodes, Map<String, Value> constants, java.util.Set<String> keep, int[] fresh) {
        Map<String, Node> producer = new HashMap<>();
        Map<String, Integer> readers = new HashMap<>();
        for (String o : keep) {
            readers.merge(o, 1, Integer::sum);
        }
        for (Node n : nodes) {
            for (String o : n.outputs()) {
                producer.put(o, n);
            }
            for (String in : n.inputs()) {
                readers.merge(in, 1, Integer::sum);
            }
        }
        java.util.Set<Node> dropped = new java.util.HashSet<>();
        List<Node> out = new ArrayList<>();
        for (Node n : nodes) {
            boolean conv = n.opType().equals("Conv") && n.attrInt("group", 1) == 1;
            boolean product = n.opType().equals("MatMul") || conv;
            Node da = product ? producer.get(n.input(0)) : null;
            Node dw = product ? producer.get(n.input(1)) : null;
            if (da == null || dw == null || !da.opType().equals("DequantizeLinear") || !dw.opType().equals("DequantizeLinear") || constants.containsKey(da.input(0))
                    || !scalar(constants, da.input(1)) || !(da.input(2).isEmpty() || scalar(constants, da.input(2))) || da.attrInt("block_size", 0) != 0
                    || !(conv ? int8Kernel(constants, dw.input(0)) : int8Weight(constants, dw.input(0))) || !zero(constants, dw.input(2)) || dw.attrInt("block_size", 0) != 0
                    || !constants.containsKey(dw.input(1)) || !(conv ? channelScale(constants, dw) : columnScale(constants, dw))) {
                out.add(n);
                continue;
            }
            Node q = producer.get(da.input(0));
            boolean fromX = q != null && q.opType().equals("QuantizeLinear") && q.input(1).equals(da.input(1)) && q.input(2).equals(da.input(2))
                    && q.attrInt("block_size", 0) == 0 && !q.attributes().containsKey("output_dtype");
            String a = fromX ? q.input(0) : da.input(0);
            out.add(conv ? int8Conv(n, a, da.input(1), da.input(2), !fromX, dw.input(0), dw.input(1), constants, fresh)
                    : int8MatMul(n, a, da.input(1), da.input(2), !fromX, dw.input(0), dw.input(1), constants, n.outputs(), fresh));
            if (readers.get(da.outputs().get(0)) == 1) {
                dropped.add(da);
                if (fromX && readers.get(q.outputs().get(0)) == 1) {
                    dropped.add(q);
                }
            }
            if (readers.get(dw.outputs().get(0)) == 1) {
                dropped.add(dw);
            }
        }
        out.removeIf(dropped::contains);
        return out;
    }

    /**
     * An Int8MatMul node: a (FP32 values to quantise with scale s and zero point z, or the integers
     * themselves when {@code integers}) times the constant INT8 weight w [K, N] with scale sw, as
     * prepared {@code Int8.Weights} constants.
     */
    private static Node int8MatMul(Node from, String a, String s, String z, boolean integers, String w, String sw, Map<String, Value> constants, List<String> outputs,
            int[] fresh) {
        Value wq = constants.get(w);
        int k = (int) wq.dims[0];
        int cols = (int) wq.dims[1];
        ai.beestream.jitnn.ops.Int8.Weights prepared = ai.beestream.jitnn.ops.Int8.Weights.ofColumns(wq.ints, k, cols, floats(constants.get(sw)));
        String base = name(from, "i8", fresh);
        constants.put(base + "w", Value.of(prepared.packed(), OnnxModel.INT32));
        constants.put(base + "s", Value.of(prepared.scale(), OnnxModel.FLOAT));
        constants.put(base + "sums", Value.of(prepared.rowSums(), OnnxModel.INT32));
        Map<String, Attribute> attrs = Map.of("integers", new Attribute("integers", ATTR_INT, 0, integers ? 1 : 0, null, null, null, null, null));
        return node(from, "Int8MatMul", List.of(a, s, z, base + "w", base + "s", base + "sums"), outputs, attrs);
    }

    /**
     * Folds what surrounds an Int8MatMul into it, as dynamic quantisation writes the graph:
     * <ul>
     * <li>its integers from DynamicQuantizeLinear(x): the product quantises x itself with that
     * scale and zero point, and DynamicQuantizeLinear computes only those when nothing else reads
     * its integers;</li>
     * <li>Cast to float then Mul by (x scale * weight scale): the weight scale goes into the
     * product's per-column scale;</li>
     * <li>Add of a constant bias [N]: the product's bias.</li>
     * </ul>
     */
    private static List<Node> int8Epilogues(List<Node> nodes, Map<String, Value> constants, java.util.Set<String> keep, int[] fresh) {
        Map<String, Node> producer = new HashMap<>();
        Map<String, List<Node>> readers = new HashMap<>();
        // a graph output counts as a reader: it stays as it is
        Node graph = new Node("#outputs", "#outputs", "", new ArrayList<>(keep), List.of(), Map.of());
        for (String o : keep) {
            readers.computeIfAbsent(o, k -> new ArrayList<>()).add(graph);
        }
        for (Node n : nodes) {
            for (String o : n.outputs()) {
                producer.put(o, n);
            }
            for (String in : n.inputs()) {
                readers.computeIfAbsent(in, k -> new ArrayList<>()).add(n);
            }
        }
        Map<Node, Node> replace = new HashMap<>();
        java.util.Set<Node> dropped = new java.util.HashSet<>();
        for (Node m : nodes) {
            if (!m.opType().equals("Int8MatMul")) {
                continue;
            }
            List<String> in = new ArrayList<>(m.inputs());
            boolean integers = m.attrInt("integers", 0) != 0;
            List<String> outputs = m.outputs();
            Node d = producer.get(in.get(0));
            if (integers && d != null && d.opType().equals("DynamicQuantizeLinear") && d.outputs().get(0).equals(in.get(0)) && d.outputs().get(2).equals(in.get(2))) {
                in.set(0, d.input(0));
                in.set(1, d.outputs().get(1));
                integers = false;
            }
            // Cast then Mul by (activation scale * constant weight scale)
            List<Node> r = readers.getOrDefault(outputs.get(0), List.of());
            if (!integers && r.size() == 1 && r.get(0).opType().equals("Cast") && r.get(0).attrInt("to", 0) == OnnxModel.FLOAT) {
                Node cast = r.get(0);
                List<Node> r2 = readers.getOrDefault(cast.outputs().get(0), List.of());
                Node mul = r2.size() == 1 && r2.get(0).opType().equals("Mul") ? r2.get(0) : null;
                String other = mul == null ? null : mul.input(0).equals(cast.outputs().get(0)) ? mul.input(1) : mul.input(0);
                Node scales = other == null ? null : producer.get(other);
                if (scales != null && scales.opType().equals("Mul") && (scales.input(0).equals(in.get(1)) || scales.input(1).equals(in.get(1)))) {
                    String ws = scales.input(0).equals(in.get(1)) ? scales.input(1) : scales.input(0);
                    Value sw = constants.get(in.get(4));
                    int cols = sw.tensor.dim(0);
                    Value wv = constants.get(ws);
                    if (wv != null && (wv.numel() == 1 || wv.numel() == cols)) {
                        float[] old = floats(sw);
                        float[] add = floats(wv);
                        float[] scaled = new float[cols];
                        for (int j = 0; j < cols; j++) {
                            scaled[j] = old[old.length == 1 ? 0 : j] * add[add.length == 1 ? 0 : j];
                        }
                        String name = name(m, "scale", fresh);
                        constants.put(name, Value.of(Tensor.persistent(Tensor.of(scaled, cols)), OnnxModel.FLOAT));
                        in.set(4, name);
                        outputs = mul.outputs();
                        dropped.add(cast);
                        dropped.add(mul);
                        if (readers.get(scales.outputs().get(0)).size() == 1) {
                            dropped.add(scales);
                        }
                    }
                }
            }
            // Add of a constant bias [N]
            List<Node> r3 = readers.getOrDefault(outputs.get(0), List.of());
            if (!integers && r3.size() == 1 && r3.get(0).opType().equals("Add")) {
                Node add = r3.get(0);
                String b = add.input(0).equals(outputs.get(0)) ? add.input(1) : add.input(0);
                Value bv = constants.get(b);
                if (bv != null && bv.numel() == constants.get(in.get(4)).tensor.dim(0) && bv.rank() <= 2) {
                    in.add(b);
                    outputs = add.outputs();
                    dropped.add(add);
                }
            }
            if (!in.equals(m.inputs()) || outputs != m.outputs()) {
                Map<String, Attribute> attrs = Map.of("integers", new Attribute("integers", ATTR_INT, 0, integers ? 1 : 0, null, null, null, null, null));
                replace.put(m, new Node(m.name(), "Int8MatMul", "", in, outputs, attrs));
            }
        }
        // DynamicQuantizeLinear whose integers nothing reads any more: scale and zero point only
        java.util.Set<String> stillRead = new java.util.HashSet<>(keep);
        for (Node n : nodes) {
            if (dropped.contains(n)) {
                continue;
            }
            stillRead.addAll(replace.getOrDefault(n, n).inputs());
        }
        List<Node> out = new ArrayList<>();
        for (Node n : nodes) {
            if (dropped.contains(n)) {
                continue;
            }
            if (n.opType().equals("DynamicQuantizeLinear") && !stillRead.contains(n.outputs().get(0))) {
                List<String> o = List.of(name(n, "unused", fresh), n.outputs().get(1), n.outputs().get(2));
                out.add(new Node(n.name(), "DynamicQuantizeParams", "", n.inputs(), o, Map.of()));
                continue;
            }
            out.add(replace.getOrDefault(n, n));
        }
        return out;
    }

    /**
     * An Int8Conv node: Conv's attributes, its input a (as {@link #int8MatMul}), the constant INT8
     * kernel w [Cout, Cin, kh, kw] with per-output-channel scale sw prepared as weights [Cout, Cin kh kw],
     * and Conv's bias when it has one.
     */
    private static Node int8Conv(Node conv, String a, String s, String z, boolean integers, String w, String sw, Map<String, Value> constants, int[] fresh) {
        Value wq = constants.get(w);
        int cout = (int) wq.dims[0];
        ai.beestream.jitnn.ops.Int8.Weights prepared = ai.beestream.jitnn.ops.Int8.Weights.ofConv(wq.ints, cout, (int) wq.dims[1], (int) wq.dims[2], (int) wq.dims[3],
                floats(constants.get(sw)));
        String base = name(conv, "i8", fresh);
        constants.put(base + "w", Value.of(prepared.packed(), OnnxModel.INT32));
        constants.put(base + "s", Value.of(prepared.scale(), OnnxModel.FLOAT));
        constants.put(base + "sums", Value.of(prepared.rowSums(), OnnxModel.INT32));
        Map<String, Attribute> attrs = new HashMap<>(conv.attributes());
        attrs.put("integers", new Attribute("integers", ATTR_INT, 0, integers ? 1 : 0, null, null, null, null, null));
        attrs.put("kernel_shape", new Attribute("kernel_shape", ATTR_INTS, 0, 0, null, null, null, new long[] { wq.dims[2], wq.dims[3] }, null));
        List<String> in = new ArrayList<>(List.of(a, s, z, base + "w", base + "s", base + "sums"));
        if (conv.inputs().size() > 2 && !conv.input(2).isEmpty()) {
            in.add(conv.input(2));
        }
        return node(conv, "Int8Conv", in, conv.outputs(), attrs);
    }

    /** A constant 4-D INT8 tensor: a 2-D convolution's kernel. */
    private static boolean int8Kernel(Map<String, Value> constants, String name) {
        Value v = constants.get(name);
        return v != null && v.isHost() && v.type == OnnxModel.INT8 && v.rank() == 4;
    }

    /** The kernel's scale is one value, or one per output channel (axis 0). */
    private static boolean channelScale(Map<String, Value> constants, Node dw) {
        Value s = constants.get(dw.input(1));
        int cout = (int) constants.get(dw.input(0)).dims[0];
        return s.numel() == 1 || s.rank() == 1 && s.numel() == cout && OnnxOps.axis(dw.attrInt("axis", 1), 4) == 0;
    }

    /** A constant 2-D INT8 tensor (UINT8 weights are not taken: the kernels multiply signed values). */
    private static boolean int8Weight(Map<String, Value> constants, String name) {
        Value v = constants.get(name);
        return v != null && v.isHost() && v.type == OnnxModel.INT8 && v.rank() == 2;
    }

    /** No zero point, or a constant one of zeros. */
    private static boolean zero(Map<String, Value> constants, String name) {
        if (name.isEmpty()) {
            return true;
        }
        Value v = constants.get(name);
        if (v == null) {
            return false;
        }
        for (float f : floats(v)) {
            if (f != 0) {
                return false;
            }
        }
        return true;
    }

    /** The weight's scale is one value, or one per column (axis 1 of the [K, N] weight). */
    private static boolean columnScale(Map<String, Value> constants, Node dw) {
        Value s = constants.get(dw.input(1));
        int cols = (int) constants.get(dw.input(0)).dims[1];
        return s.numel() == 1 || s.rank() == 1 && s.numel() == cols && OnnxOps.axis(dw.attrInt("axis", 1), 2) == 1;
    }

    /**
     * QuantizeLinear then DequantizeLinear with the same per-tensor scale and zero point, on an
     * activation: one FakeQuantize node (one pass, and fusable into the producer's epilogue). The
     * QuantizeLinear stays when something else reads its integers.
     */
    private static List<Node> fakeQuantize(List<Node> nodes, Map<String, Value> constants) {
        Map<String, Node> quantizer = new HashMap<>();
        Map<String, Integer> readers = new HashMap<>();
        for (Node n : nodes) {
            if (n.opType().equals("QuantizeLinear")) {
                quantizer.put(n.outputs().get(0), n);
            }
            for (String in : n.inputs()) {
                readers.merge(in, 1, Integer::sum);
            }
        }
        Map<Node, Integer> replaced = new HashMap<>();
        List<Node> out = new ArrayList<>();
        for (Node n : nodes) {
            Node q = n.opType().equals("DequantizeLinear") ? quantizer.get(n.input(0)) : null;
            if (q != null && !constants.containsKey(q.input(0)) && n.input(1).equals(q.input(1)) && n.input(2).equals(q.input(2)) && scalar(constants, q.input(1))
                    && (q.input(2).isEmpty() || scalar(constants, q.input(2))) && q.attrInt("block_size", 0) == 0) {
                out.add(new Node(n.name(), "FakeQuantize", "", q.inputs(), n.outputs(), q.attributes()));
                replaced.merge(q, 1, Integer::sum);
            } else {
                out.add(n);
            }
        }
        // drop the quantizers whose every reader became a FakeQuantize
        out.removeIf(n -> replaced.containsKey(n) && replaced.get(n).equals(readers.getOrDefault(n.outputs().get(0), 0)));
        return out;
    }

    private static boolean scalar(Map<String, Value> constants, String name) {
        Value v = constants.get(name);
        long n = 1;
        if (v == null) {
            return false;
        }
        for (long d : v.dims) {
            n *= d;
        }
        return n == 1;
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
        h.put("Int8Conv", (n, in) -> {
            Tensor x = OnnxOps.t(in, 0);
            if (x.rank() != 4) {
                throw new UnsupportedOperationException(n.name() + ": INT8 convolution of a " + x.shape() + " input (2-D only)");
            }
            boolean integers = n.attrInt("integers", 0) != 0;
            int type = OnnxOps.present(in, 2) ? in.get(2).type : integers ? in.get(0).type : OnnxModel.UINT8;
            boolean unsigned = type != OnnxModel.INT8;
            Tensor s = in.get(1).tensor(true).reshape(1);
            Tensor z = OnnxOps.present(in, 2) ? in.get(2).tensor(true).reshape(1) : Tensor.of(new float[] { 0 }, 1);
            long[] kernel = n.attrInts("kernel_shape", null);
            long[] strides = n.attrInts("strides", new long[] { 1, 1 });
            long[] dil = n.attrInts("dilations", new long[] { 1, 1 });
            long[] pads = OnnxOps.pads(n, 2, new long[] { x.dim(2), x.dim(3) }, kernel, strides, dil);
            Tensor cout = in.get(4).tensor;
            ai.beestream.jitnn.ops.Int8.Weights w = new ai.beestream.jitnn.ops.Int8.Weights(in.get(3).tensor, cout, in.get(5).tensor, cout.dim(0),
                    (x.dim(1) + 3) / 4 * 4 * (int) kernel[0] * (int) kernel[1]);
            Tensor bias = OnnxOps.present(in, 6) ? in.get(6).tensor(true).reshape(w.n()) : null;
            if (integers) {
                // integers given: back to their values, so the padding becomes the zero point again
                x = x.sub(z.reshape()).mul(s.reshape());
            }
            Tensor y = ai.beestream.jitnn.ops.Int8.conv2d(x, s, z, unsigned, w, bias, (int) kernel[0], (int) kernel[1],
                    new int[] { (int) strides[0], (int) strides[1] }, new int[] { (int) pads[0], (int) pads[1], (int) pads[2], (int) pads[3] }, new int[] { (int) dil[0], (int) dil[1] });
            return List.of(Value.of(y, OnnxModel.FLOAT));
        });
        h.put("Int8MatMul", (n, in) -> {
            Tensor x = OnnxOps.t(in, 0);
            Tensor s = in.get(1).tensor(true).reshape(1);
            Tensor z = OnnxOps.present(in, 2) ? in.get(2).tensor(true).reshape(1) : Tensor.of(new float[] { 0 }, 1);
            boolean integers = n.attrInt("integers", 0) != 0;
            // the activation's integer type: its zero point's, else that of the integers given, else UINT8
            int type = OnnxOps.present(in, 2) ? in.get(2).type : integers ? in.get(0).type : OnnxModel.UINT8;
            boolean unsigned = type != OnnxModel.INT8;
            int k = x.dim(-1);
            ai.beestream.jitnn.ops.Int8.Weights w = new ai.beestream.jitnn.ops.Int8.Weights(in.get(3).tensor, in.get(4).tensor, in.get(5).tensor, in.get(4).tensor.dim(0), k);
            Tensor flat = x.reshape(-1, k);
            Tensor q = integers ? ai.beestream.jitnn.ops.Int8.quantize(flat, Tensor.of(new float[] { 1 }, 1), Tensor.of(new float[] { 0 }, 1), unsigned)
                    : ai.beestream.jitnn.ops.Int8.quantize(flat, s, z, unsigned);
            Tensor bias = OnnxOps.present(in, 6) ? in.get(6).tensor(true).reshape(w.n()) : null;
            Tensor y = ai.beestream.jitnn.ops.Int8.linear(q, s, z, unsigned, w, bias);
            int[] dims = x.shape().dims();
            dims[dims.length - 1] = w.n();
            return List.of(Value.of(y.reshape(dims), OnnxModel.FLOAT));
        });
        h.put("FakeQuantize", (n, in) -> {
            int type = quantType(n, in);
            float[] range = range(n, type);
            float scale = floats(in.get(1))[0];
            float zp = OnnxOps.present(in, 2) ? floats(in.get(2))[0] : 0.0f;
            int bits = range[1] - range[0] == 255.0f ? 8 : 16;
            return List.of(Value.of(F.fakeQuantize(OnnxOps.t(in, 0), scale, range[0] - zp, bits), OnnxModel.FLOAT));
        });
        h.put("DequantizeLinear", (n, in) -> List.of(dequantize(n, in)));
        h.put("QuantizeLinear", (n, in) -> {
            int type = quantType(n, in);
            noBlocks(n);
            float[] range = range(n, type);
            Tensor x = OnnxOps.t(in, 0);
            Tensor q = x.div(perAxis(n, in.get(1), x.rank())).round();
            if (OnnxOps.present(in, 2)) {
                q = q.add(perAxis(n, in.get(2), x.rank()));
            }
            return List.of(Value.of(q.clamp(range[0], range[1]), type));
        });
        h.put("DynamicQuantizeParams", (n, in) -> {
            List<Value> v = dynamicQuantize(OnnxOps.t(in, 0), false);
            return List.of(Value.of(Tensor.of(new float[] { 0 }, 1), OnnxModel.FLOAT), v.get(1), v.get(2));
        });
        h.put("DynamicQuantizeLinear", (n, in) -> dynamicQuantize(OnnxOps.t(in, 0), true));
    }

    /** DynamicQuantizeLinear: the UINT8 values of x (when {@code values}), its scale and its zero point. */
    private static List<Value> dynamicQuantize(Tensor x, boolean values) {
        Tensor flat = x.reshape(x.numel());
        Tensor zero = Tensor.scalar(0.0f);
        Tensor lo = flat.min(0, false).minimum(zero);
        Tensor hi = flat.max(0, false).maximum(zero);
        Tensor scale = hi.sub(lo).div(Tensor.scalar(255.0f));
        Tensor zp = zero.sub(lo.div(scale)).round().clamp(0, 255);
        Tensor y = values ? x.div(scale).round().add(zp).clamp(0, 255) : null;
        return java.util.Arrays.asList(Value.of(y == null ? zp : y, OnnxModel.UINT8), Value.of(scale, OnnxModel.FLOAT), Value.of(zp, OnnxModel.UINT8));
    }

    /** The integer type QuantizeLinear produces: output_dtype, else the zero point's, else UINT8. */
    private static int quantType(Node n, List<Value> in) {
        int type = (int) n.attrInt("output_dtype", OnnxOps.present(in, 2) ? in.get(2).type : OnnxModel.UINT8);
        // a zero point created by the load-time rewriting carries no element type of its own
        return type == OnnxModel.INT64 ? OnnxModel.UINT8 : type;
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
