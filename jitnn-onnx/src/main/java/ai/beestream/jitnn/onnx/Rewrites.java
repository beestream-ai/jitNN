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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.onnx.OnnxModel.Node;

/**
 * Load-time rewrites of exported graphs into the forms jitNN's fused kernels serve:
 * <ul>
 * <li>MatMul by a constant matrix W [K, N] (and an Add of a constant bias [N]) becomes one linear
 * layer with the weight stored transposed, [N, K], as F.linear takes it (one GEMM with the bias in
 * its epilogue, where jitNN's inference fusions can also fold an activation).
 * <li>The erf form of GELU that exporters write, x * 0.5 * (1 + erf(x / sqrt 2)), becomes one GELU.
 * <li>Attention written as MatMul(q, k^T) [+ mask] -> Softmax(last axis) -> MatMul(., v) becomes
 * one call to F.scaledDotProductAttention (jitNN's fused attention), with constant scalings of q
 * and k^T folded into its scale and the exporter's Reshape-Transpose-Reshape of k undone.
 * </ul>
 * With {@code half}, the linear layers' weights are stored in FP16.
 * Only nodes whose intermediate results have no other consumer are rewritten.
 */
final class Rewrites {

    static final String LINEAR = "jitnn.Linear";
    static final String ATTENTION = "jitnn.Attention";

    private Rewrites() {
    }

    static List<Node> apply(List<Node> nodes, Map<String, Value> constants, Set<String> graphOutputs, boolean half) {
        Map<String, Integer> uses = new HashMap<>();
        Map<String, Node> producer = new HashMap<>();
        for (Node n : nodes) {
            for (String i : n.inputs()) {
                uses.merge(i, 1, Integer::sum);
            }
            for (String o : n.outputs()) {
                producer.put(o, n);
            }
        }
        graphOutputs.forEach(o -> uses.merge(o, 1, Integer::sum));
        Map<String, Node> consumer = new HashMap<>();
        for (Node n : nodes) {
            for (String i : n.inputs()) {
                if (uses.getOrDefault(i, 0) == 1) {
                    consumer.put(i, n);
                }
            }
        }
        Set<Node> dropped = new HashSet<>();
        Map<Node, Node> replaced = new HashMap<>();
        for (Node n : nodes) {
            if (n.opType().equals("MatMul") && n.inputs().size() == 2) {
                Value w = constants.get(n.input(1));
                if (w == null || w.tensor == null || w.rank() != 2) {
                    continue;
                }
                String out = n.outputs().get(0);
                String bias = null;
                Node add = consumer.get(out);
                if (add != null && add.opType().equals("Add")) {
                    String other = add.input(0).equals(out) ? add.input(1) : add.input(0);
                    Value b = constants.get(other);
                    if (b != null && b.tensor != null && b.rank() == 1 && b.dims[0] == w.dims[1]) {
                        bias = other;
                    }
                }
                String wt = n.input(1) + "#transposed";
                if (!constants.containsKey(wt)) {
                    Tensor t = w.tensor.transpose(0, 1).mul(1.0f);
                    constants.put(wt, Value.of(half ? Tensor.persistentHalf(t.toFloatArray(), t.shape().dims()) : Tensor.persistent(t)));
                }
                List<String> ins = bias == null ? List.of(n.input(0), wt) : List.of(n.input(0), wt, bias);
                List<String> outs = bias == null ? n.outputs() : add.outputs();
                replaced.put(n, new Node(n.name(), LINEAR, "", ins, outs, Map.of()));
                if (bias != null) {
                    dropped.add(add);
                }
            }
            if (n.opType().equals("Softmax") && !replaced.containsKey(n)) {
                attention(n, producer, consumer, constants, dropped, replaced);
            }
            if (n.opType().equals("Erf")) {
                Node gelu = geluRoot(n, producer, consumer, constants, dropped);
                if (gelu != null) {
                    replaced.put(n, gelu);
                }
            }
        }
        // a node both dropped and replaced keeps its replacement; dropped producers that some other
        // node still reads are kept (the matcher only removes single-use intermediates)
        List<Node> out = new ArrayList<>();
        for (Node n : nodes) {
            if (dropped.contains(n) && !replaced.containsKey(n)) {
                continue;
            }
            out.add(replaced.getOrDefault(n, n));
        }
        return out;
    }

    private static float scalarConst(Map<String, Value> constants, String name) {
        Value v = constants.get(name);
        float[] f = v == null ? null : v.hostFloats();
        return f != null && f.length == 1 ? f[0] : Float.NaN;
    }

    /** x when {@code name} is Mul(x, c) with a constant scalar c (single use): [x, c]; else [name, 1]. */
    private static Object[] unscale(String name, Map<String, Node> producer, Map<String, Node> consumer, Map<String, Value> constants, List<Node> used) {
        Node m = producer.get(name);
        if (m != null && m.opType().equals("Mul") && consumer.containsKey(name)) {
            float c1 = scalarConst(constants, m.input(1));
            float c0 = scalarConst(constants, m.input(0));
            if (!Float.isNaN(c1)) {
                used.add(m);
                return new Object[] { m.input(0), c1 };
            }
            if (!Float.isNaN(c0)) {
                used.add(m);
                return new Object[] { m.input(1), c0 };
            }
        }
        return new Object[] { name, 1.0f };
    }

    /**
     * Matches MatMul(a, b) [-> Add(mask)] -> Softmax(axis -1) -> MatMul(p, v), every intermediate
     * single-use; a = q [* c], b = k^T [* c]. k^T written as Reshape(Transpose(Reshape(k), [0, 2, 1]))
     * is traced back to k (form 1), else b is used transposed (form 0). The attention node (attributes
     * scale and form) takes the second MatMul's place, so v is available there.
     */
    private static void attention(Node softmax, Map<String, Node> producer, Map<String, Node> consumer, Map<String, Value> constants, Set<Node> dropped,
            Map<Node, Node> replaced) {
        if (softmax.attrInt("axis", -1) != -1) {
            return;
        }
        String sIn = softmax.input(0);
        if (consumer.get(sIn) != softmax) {
            return;
        }
        Node before = producer.get(sIn);
        String mask = null;
        Node add = null;
        if (before != null && before.opType().equals("Add")) {
            add = before;
            Node l = producer.get(before.input(0));
            Node r = producer.get(before.input(1));
            if (l != null && l.opType().equals("MatMul") && consumer.get(before.input(0)) == before) {
                mask = before.input(1);
                before = l;
            } else if (r != null && r.opType().equals("MatMul") && consumer.get(before.input(1)) == before) {
                mask = before.input(0);
                before = r;
            } else {
                return;
            }
        }
        if (before == null || !before.opType().equals("MatMul") || replaced.containsKey(before)) {
            return;
        }
        Node pv = consumer.get(softmax.outputs().get(0));
        if (pv == null || !pv.opType().equals("MatMul") || !pv.input(0).equals(softmax.outputs().get(0))) {
            return;
        }
        List<Node> used = new ArrayList<>();
        Object[] qa = unscale(before.input(0), producer, consumer, constants, used);
        Object[] kb = unscale(before.input(1), producer, consumer, constants, used);
        float scale = (float) qa[1] * (float) kb[1];
        String kt = (String) kb[0];
        String k = kt;
        int form = 0;
        Node r2 = producer.get(kt);
        if (r2 != null && r2.opType().equals("Reshape") && consumer.get(kt) != null) {
            Node tr = producer.get(r2.input(0));
            if (tr != null && tr.opType().equals("Transpose") && java.util.Arrays.equals(tr.attrInts("perm", null), new long[] { 0, 2, 1 })) {
                Node r1 = producer.get(tr.input(0));
                if (r1 != null && r1.opType().equals("Reshape")) {
                    k = r1.input(0);
                    form = 1;
                    used.add(r2);
                    used.add(tr);
                    used.add(r1);
                }
            }
        }
        dropped.add(before);
        dropped.add(softmax);
        if (add != null) {
            dropped.add(add);
        }
        for (Node u : used) {
            dropped.add(u);
        }
        Map<String, OnnxModel.Attribute> attrs = new java.util.LinkedHashMap<>();
        attrs.put("scale", new OnnxModel.Attribute("scale", 1, scale, 0, null, null, null, null, null));
        attrs.put("form", new OnnxModel.Attribute("form", 2, 0, form, null, null, null, null, null));
        List<String> ins = new ArrayList<>(List.of((String) qa[0], k, pv.input(1)));
        if (mask != null) {
            ins.add(mask);
        }
        replaced.put(pv, new Node(softmax.name(), ATTENTION, "", ins, pv.outputs(), attrs));
    }

    private static boolean isConst(Map<String, Value> constants, String name, double value) {
        Value v = constants.get(name);
        if (v == null) {
            return false;
        }
        float[] f = v.hostFloats();
        return f != null && f.length == 1 && Math.abs(f[0] - value) < 1e-6 * Math.max(1, Math.abs(value));
    }

    private static String other(Node n, String name) {
        return n.input(0).equals(name) ? n.input(1) : n.input(0);
    }

    /**
     * Matches Div(x, sqrt2) | Mul(x, 1/sqrt2) -> Erf -> Add(1) -> Mul(0.5) -> Mul(x) (the two Muls in
     * either order); returns a Gelu node producing the last Mul's output, dropping the others.
     */
    private static Node geluRoot(Node erf, Map<String, Node> producer, Map<String, Node> consumer, Map<String, Value> constants, Set<Node> dropped) {
        Node scale = producer.get(erf.input(0));
        if (scale == null || consumer.get(erf.input(0)) != erf) {
            return null;
        }
        String x;
        if (scale.opType().equals("Div") && isConst(constants, scale.input(1), Math.sqrt(2))) {
            x = scale.input(0);
        } else if (scale.opType().equals("Mul") && (isConst(constants, scale.input(1), 1 / Math.sqrt(2)) || isConst(constants, scale.input(0), 1 / Math.sqrt(2)))) {
            x = isConst(constants, scale.input(1), 1 / Math.sqrt(2)) ? scale.input(0) : scale.input(1);
        } else {
            return null;
        }
        Node add = consumer.get(erf.outputs().get(0));
        if (add == null || !add.opType().equals("Add") || !isConst(constants, other(add, erf.outputs().get(0)), 1.0)) {
            return null;
        }
        Node m1 = consumer.get(add.outputs().get(0));
        if (m1 == null || !m1.opType().equals("Mul")) {
            return null;
        }
        String o1 = other(m1, add.outputs().get(0));
        Node m2 = consumer.get(m1.outputs().get(0));
        if (m2 == null || !m2.opType().equals("Mul")) {
            return null;
        }
        String o2 = other(m2, m1.outputs().get(0));
        boolean half1 = isConst(constants, o1, 0.5);
        boolean half2 = isConst(constants, o2, 0.5);
        if (!((half1 && o2.equals(x)) || (half2 && o1.equals(x)))) {
            return null;
        }
        dropped.add(scale);
        dropped.add(add);
        dropped.add(m1);
        dropped.add(m2);
        return new Node(erf.name(), "Gelu", "", List.of(x), m2.outputs(), Map.of());
    }
}
