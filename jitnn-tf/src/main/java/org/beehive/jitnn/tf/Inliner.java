package org.beehive.jitnn.tf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.beehive.jitnn.tf.TfGraph.Function;
import org.beehive.jitnn.tf.TfGraph.Node;

/**
 * Flattens TensorFlow 2 graphs: every PartitionedCall / StatefulPartitionedCall is replaced by its
 * function's body, recursively, with node names prefixed by the call's name. Tensor references are
 * rewritten to the graph form "node:index" (the function form "node:output_arg:index" becomes a
 * flat output index). Identity-like calls to functions' arguments resolve straight to the caller's
 * inputs.
 */
final class Inliner {

    /** Offsets of named output arguments for ops with more than one output argument. */
    private static final Map<String, Map<String, Integer>> OUTPUT_ARGS = Map.of(
            "TopKV2", Map.of("values", 0, "indices", 1),
            "FusedBatchNorm", Map.of("y", 0, "batch_mean", 1, "batch_variance", 2, "reserve_space_1", 3, "reserve_space_2", 4),
            "FusedBatchNormV2", Map.of("y", 0, "batch_mean", 1, "batch_variance", 2, "reserve_space_1", 3, "reserve_space_2", 4),
            "FusedBatchNormV3", Map.of("y", 0, "batch_mean", 1, "batch_variance", 2, "reserve_space_1", 3, "reserve_space_2", 4, "reserve_space_3", 5),
            "Unique", Map.of("y", 0, "idx", 1));

    private final Map<String, Function> functions;
    private final List<Node> out = new ArrayList<>();
    /** Graph-level aliases: a call's output "call:i" to the tensor it returns. */
    private final Map<String, String> alias = new HashMap<>();
    private int depth;

    private Inliner(Map<String, Function> functions) {
        this.functions = functions;
    }

    static boolean isCall(Node n) {
        return (n.op().equals("PartitionedCall") || n.op().equals("StatefulPartitionedCall")) && n.attr("f") != null;
    }

    /** The flat node list, and aliases for the outputs of the calls that were inlined. */
    record Result(List<Node> nodes, Map<String, String> alias) {

        /** A reference with call outputs followed to the tensors they return. */
        String resolve(String ref) {
            String r = TfGraphs.canonical(ref);
            for (int guard = 0; alias.containsKey(r) && guard < 1000; guard++) {
                r = alias.get(r);
            }
            return r;
        }
    }

    static Result inline(List<Node> graph, Map<String, Function> functions) {
        Inliner in = new Inliner(functions);
        for (Node n : graph) {
            in.add(n, "", Map.of(), null);
        }
        return new Result(in.out, in.alias);
    }

    /**
     * Adds node {@code n} of a function body (or of the graph, with an empty prefix). {@code args}
     * maps the function's argument names to graph references; {@code body} names the body's nodes,
     * for reading their output-argument names.
     */
    private void add(Node n, String prefix, Map<String, String> args, Map<String, Node> body) {
        List<String> inputs = new ArrayList<>();
        for (String ref : n.inputs()) {
            if (ref.startsWith("^")) {
                continue;
            }
            inputs.add(body == null ? TfGraphs.canonical(ref) : fromFunction(ref, prefix, args, body));
        }
        String name = prefix + n.name();
        if (isCall(n)) {
            Function f = functions.get(n.attr("f").func());
            if (f == null) {
                throw new UnsupportedOperationException("TensorFlow call " + name + " to an unknown function " + n.attr("f").func());
            }
            if (++depth > 64) {
                throw new IllegalStateException("TensorFlow functions nested too deep at " + name);
            }
            Map<String, String> fargs = new HashMap<>();
            for (int i = 0; i < f.inputArgs().size() && i < inputs.size(); i++) {
                fargs.put(f.inputArgs().get(i), inputs.get(i));
            }
            Map<String, Node> fbody = new HashMap<>();
            for (Node b : f.nodes()) {
                fbody.put(b.name(), b);
            }
            String p = name + "/";
            for (Node b : f.nodes()) {
                add(b, p, fargs, fbody);
            }
            for (int i = 0; i < f.outputArgs().size(); i++) {
                String ret = f.ret().get(f.outputArgs().get(i));
                alias.put(name + ":" + i, fromFunction(ret, p, fargs, fbody));
            }
            depth--;
            return;
        }
        out.add(new Node(name, n.op(), inputs, n.attrs()));
    }

    /** A function-body reference ("arg", "node:out_arg:i" or "node") as a graph reference. */
    private String fromFunction(String ref, String prefix, Map<String, String> args, Map<String, Node> body) {
        String[] parts = ref.split(":");
        if (parts.length == 1) {
            String a = args.get(ref);
            return a != null ? a : prefix + ref + ":0";
        }
        Node producer = body.get(parts[0]);
        int index = parts.length == 3 ? Integer.parseInt(parts[2]) : Integer.parseInt(parts[1]);
        if (parts.length == 3 && producer != null) {
            Map<String, Integer> offsets = OUTPUT_ARGS.get(producer.op());
            if (offsets != null && offsets.containsKey(parts[1])) {
                index += offsets.get(parts[1]);
            }
        }
        return prefix + parts[0] + ":" + index;
    }
}
