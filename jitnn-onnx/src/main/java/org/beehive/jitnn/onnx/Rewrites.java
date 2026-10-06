package org.beehive.jitnn.onnx;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.onnx.OnnxModel.Node;

/**
 * Load-time rewrites of exported graphs into the forms jitNN's fused kernels serve:
 * <ul>
 * <li>MatMul by a constant matrix W [K, N] (and an Add of a constant bias [N]) becomes one linear
 * layer with the weight stored transposed, [N, K], as F.linear takes it (one GEMM with the bias in
 * its epilogue, where jitNN's inference fusions can also fold an activation).
 * <li>The erf form of GELU that exporters write, x * 0.5 * (1 + erf(x / sqrt 2)), becomes one GELU.
 * </ul>
 * Only nodes whose intermediate results have no other consumer are rewritten.
 */
final class Rewrites {

    static final String LINEAR = "jitnn.Linear";

    private Rewrites() {
    }

    static List<Node> apply(List<Node> nodes, Map<String, Value> constants, Set<String> graphOutputs) {
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
                    constants.put(wt, Value.of(Tensor.persistent(w.tensor.transpose(0, 1).mul(1.0f))));
                }
                List<String> ins = bias == null ? List.of(n.input(0), wt) : List.of(n.input(0), wt, bias);
                List<String> outs = bias == null ? n.outputs() : add.outputs();
                replaced.put(n, new Node(n.name(), LINEAR, "", ins, outs, Map.of()));
                if (bias != null) {
                    dropped.add(add);
                }
            }
            if (n.opType().equals("Erf")) {
                Node gelu = geluRoot(n, producer, consumer, constants, dropped);
                if (gelu != null) {
                    replaced.put(n, gelu);
                }
            }
        }
        List<Node> out = new ArrayList<>();
        for (Node n : nodes) {
            if (dropped.contains(n)) {
                continue;
            }
            out.add(replaced.getOrDefault(n, n));
        }
        return out;
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
