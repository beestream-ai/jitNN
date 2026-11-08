package ai.beestream.jitnn.tf;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.onnx.OnnxSession;
import ai.beestream.jitnn.onnx.Proto;
import ai.beestream.jitnn.tf.TfGraph.Node;

/**
 * A TensorFlow model on jitNN: a SavedModel directory (TensorFlow 2 / Keras {@code export}, with
 * its variables checkpoint) or a frozen GraphDef file. The graph is flattened (function calls
 * inlined), its variables read from the checkpoint, and it is translated to the operators of
 * jitnn-onnx, which runs it.
 *
 * <pre>{@code
 * TfModel model = TfModel.load(Path.of("my_saved_model"));
 * Map<String, Tensor> out = model.run(Map.of("keras_tensor", Tensor.of(x, 1, 4)));
 * }</pre>
 *
 * Inputs and outputs carry the signature's names. Integer inputs (token ids) are INT32 tensors;
 * images stay NHWC, as TensorFlow takes them.
 */
public final class TfModel implements AutoCloseable {

    private final OnnxSession session;

    private TfModel(OnnxSession session) {
        this.session = session;
    }

    /** Loads a SavedModel directory's {@code serving_default} signature, or a frozen GraphDef file. */
    public static TfModel load(Path path) throws IOException {
        return load(path, "serving_default", false);
    }

    /**
     * Loads signature {@code signature} of a SavedModel (ignored for a frozen GraphDef, whose
     * inputs are its placeholders and whose outputs are the nodes nothing consumes). With
     * {@code halfPrecision}, matrix products run on FP16 operands, as in jitnn-onnx.
     */
    public static TfModel load(Path path, String signature, boolean halfPrecision) throws IOException {
        if (Files.isDirectory(path)) {
            return new TfModel(OnnxSession.of(savedModel(path, signature), halfPrecision));
        }
        return new TfModel(OnnxSession.of(frozen(path), halfPrecision));
    }

    /** The model's input names, as {@code run} takes them. */
    public List<String> inputNames() {
        return session.inputNames();
    }

    /** The model's output names. */
    public List<String> outputNames() {
        return session.outputNames();
    }

    /** The jitnn-onnx session the model was translated to. */
    public OnnxSession session() {
        return session;
    }

    /** Releases the model's weights (see {@link OnnxSession#close()}); the model cannot run afterwards. */
    @Override
    public void close() {
        session.close();
    }

    /** Runs the model; outputs are lazy jitNN tensors, by signature output name. */
    public Map<String, Tensor> run(Map<String, Tensor> inputs) {
        return session.run(inputs);
    }

    // ---------------------------------------------------------------- loading

    private static ByteBuffer map(Path file) throws IOException {
        try (RandomAccessFile f = new RandomAccessFile(file.toFile(), "r"); FileChannel ch = f.getChannel()) {
            return ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN);
        }
    }

    private static ai.beestream.jitnn.onnx.OnnxModel savedModel(Path dir, String signature) throws IOException {
        TfGraph.SavedModel sm = TfGraph.savedModel(new Proto(map(dir.resolve("saved_model.pb"))));
        TfGraph.Signature sig = sm.signatures().get(signature);
        if (sig == null) {
            throw new IllegalArgumentException("no signature '" + signature + "' in " + dir + "; it has " + sm.signatures().keySet());
        }
        Inliner.Result flat = Inliner.inline(sm.graph().nodes(), sm.graph().functions());
        Map<String, TfGraph.Tensor> variables = new HashMap<>();
        Path prefix = dir.resolve("variables").resolve("variables");
        if (Files.exists(Path.of(prefix + ".index"))) {
            TensorBundle bundle = new TensorBundle(prefix);
            for (Map.Entry<String, String> e : checkpointKeys(flat).entrySet()) {
                if (bundle.contains(e.getValue())) {
                    variables.put(e.getKey(), bundle.get(e.getValue()));
                }
            }
        }
        List<Node> needed = needed(flat, sig.outputs().values());
        return new Translator(new Inliner.Result(needed, flat.alias()), variables).translate(sig.inputs(), sig.outputs());
    }

    private static ai.beestream.jitnn.onnx.OnnxModel frozen(Path file) throws IOException {
        TfGraph.Graph g = TfGraph.graph(new Proto(map(file)));
        Inliner.Result flat = Inliner.inline(g.nodes(), g.functions());
        Set<String> consumed = new HashSet<>();
        for (Node n : flat.nodes()) {
            for (String in : n.data()) {
                consumed.add(TfGraphs.node(flat.resolve(in)));
            }
        }
        Map<String, String> inputs = new LinkedHashMap<>();
        Map<String, String> outputs = new LinkedHashMap<>();
        for (Node n : flat.nodes()) {
            if (n.op().equals("Placeholder")) {
                inputs.put(n.name(), n.name() + ":0");
            } else if (!consumed.contains(n.name()) && !n.op().equals("NoOp") && !n.op().equals("Const")) {
                outputs.put(n.name(), n.name() + ":0");
            }
        }
        return new Translator(new Inliner.Result(needed(flat, outputs.values()), flat.alias()), Map.of()).translate(inputs, outputs);
    }

    /** The nodes the outputs depend on, in graph order. */
    private static List<Node> needed(Inliner.Result flat, Iterable<String> outputs) {
        Map<String, Node> byName = new HashMap<>();
        for (Node n : flat.nodes()) {
            byName.put(n.name(), n);
        }
        Set<String> keep = new HashSet<>();
        ArrayDeque<String> work = new ArrayDeque<>();
        outputs.forEach(o -> work.add(TfGraphs.node(flat.resolve(o))));
        while (!work.isEmpty()) {
            String name = work.pop();
            Node n = byName.get(name);
            if (n != null && keep.add(name)) {
                for (String in : n.data()) {
                    work.add(TfGraphs.node(flat.resolve(in)));
                }
            }
        }
        List<Node> out = new ArrayList<>();
        for (Node n : flat.nodes()) {
            if (keep.contains(n.name())) {
                out.add(n);
            }
        }
        return out;
    }

    /**
     * Variable handle node to checkpoint key, read off the restore graph: each AssignVariableOp
     * writes output k of a RestoreV2 (through Identities) into a handle, and RestoreV2's second
     * input is the constant list of checkpoint keys.
     */
    private static Map<String, String> checkpointKeys(Inliner.Result flat) {
        Map<String, Node> byName = new HashMap<>();
        for (Node n : flat.nodes()) {
            byName.put(n.name(), n);
        }
        Map<String, String> keys = new HashMap<>();
        for (Node n : flat.nodes()) {
            if (!n.op().equals("AssignVariableOp") || n.data().size() < 2) {
                continue;
            }
            String value = flat.resolve(n.data().get(1));
            Node v = byName.get(TfGraphs.node(value));
            while (v != null && v.op().equals("Identity")) {
                value = flat.resolve(v.data().get(0));
                v = byName.get(TfGraphs.node(value));
            }
            if (v == null || !v.op().equals("RestoreV2")) {
                continue;
            }
            int k = Integer.parseInt(value.substring(value.lastIndexOf(':') + 1));
            Node names = byName.get(TfGraphs.node(flat.resolve(v.data().get(1))));
            if (names == null || !names.op().equals("Const") || names.attr("value").tensor().strings().size() <= k) {
                continue;
            }
            String handle = TfGraphs.node(flat.resolve(n.data().get(0)));
            keys.put(handle, names.attr("value").tensor().strings().get(k));
        }
        return keys;
    }
}
