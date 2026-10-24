package org.beehive.jitnn.onnx;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * An ONNX model as read from its file: the main graph's nodes, initializers, inputs and outputs,
 * and the operator-set versions. Initializers stored outside the model file (ONNX external data)
 * are memory-mapped from their files next to it.
 */
public final class OnnxModel {

    /** ONNX TensorProto.DataType values. */
    public static final int FLOAT = 1;
    public static final int UINT8 = 2;
    public static final int INT8 = 3;
    public static final int UINT16 = 4;
    public static final int INT16 = 5;
    public static final int INT32 = 6;
    public static final int INT64 = 7;
    public static final int BOOL = 9;
    public static final int FLOAT16 = 10;
    public static final int DOUBLE = 11;
    public static final int BFLOAT16 = 16;

    /** A tensor stored in the model: dims, element type and its data as little-endian bytes. */
    public record TensorData(String name, long[] dims, int dataType, ByteBuffer raw) {

        public int numel() {
            long n = 1;
            for (long d : dims) {
                n *= d;
            }
            return (int) n;
        }

        /** The values as floats (FLOAT, FLOAT16, BFLOAT16, DOUBLE and the integer types). */
        public float[] floats() {
            int n = numel();
            float[] out = new float[n];
            ByteBuffer b = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                out[i] = switch (dataType) {
                    case FLOAT -> b.getFloat(4 * i);
                    case DOUBLE -> (float) b.getDouble(8 * i);
                    case FLOAT16 -> Float.float16ToFloat(b.getShort(2 * i));
                    case BFLOAT16 -> Float.intBitsToFloat((b.getShort(2 * i) & 0xFFFF) << 16);
                    case INT64 -> b.getLong(8 * i);
                    case INT32 -> b.getInt(4 * i);
                    case INT8 -> b.get(i);
                    case INT16 -> b.getShort(2 * i);
                    case UINT16 -> b.getShort(2 * i) & 0xFFFF;
                    case UINT8, BOOL -> b.get(i) & 0xFF;
                    default -> throw new UnsupportedOperationException("ONNX data type " + dataType + " of " + name);
                };
            }
            return out;
        }

        /** The values as longs (INT64, INT32, INT8, UINT8, BOOL, and integral floats). */
        public long[] longs() {
            int n = numel();
            long[] out = new long[n];
            ByteBuffer b = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                out[i] = switch (dataType) {
                    case INT64 -> b.getLong(8 * i);
                    case INT32 -> b.getInt(4 * i);
                    case INT8 -> b.get(i);
                    case INT16 -> b.getShort(2 * i);
                    case UINT16 -> b.getShort(2 * i) & 0xFFFF;
                    case UINT8, BOOL -> b.get(i) & 0xFF;
                    case FLOAT -> (long) b.getFloat(4 * i);
                    case DOUBLE -> (long) b.getDouble(8 * i);
                    default -> throw new UnsupportedOperationException("ONNX data type " + dataType + " of " + name + " as integers");
                };
            }
            return out;
        }
    }

    /** An attribute of a node; which fields are set depends on its type. */
    public record Attribute(String name, int type, float f, long i, String s, TensorData t, float[] floats, long[] ints, Graph g) {
    }

    public record Node(String name, String opType, String domain, List<String> inputs, List<String> outputs, Map<String, Attribute> attributes) {

        public Attribute attr(String key) {
            return attributes.get(key);
        }

        public long attrInt(String key, long fallback) {
            Attribute a = attributes.get(key);
            return a == null ? fallback : a.i();
        }

        public float attrFloat(String key, float fallback) {
            Attribute a = attributes.get(key);
            return a == null ? fallback : a.f();
        }

        public long[] attrInts(String key, long[] fallback) {
            Attribute a = attributes.get(key);
            return a == null ? fallback : a.ints();
        }

        public String attrString(String key, String fallback) {
            Attribute a = attributes.get(key);
            return a == null ? fallback : a.s();
        }

        /** Input k, or "" when absent (ONNX marks skipped optional inputs with ""). */
        public String input(int k) {
            return k < inputs.size() ? inputs.get(k) : "";
        }
    }

    /**
     * A graph input or output: name, element type and dims (-1 for a symbolic dimension), with the
     * symbolic dimensions' names ({@code dim_param}, null where a dimension has none or dims is null).
     */
    public record ValueInfo(String name, int elemType, long[] dims, String[] dimNames) {

        public ValueInfo(String name, int elemType, long[] dims) {
            this(name, elemType, dims, null);
        }
    }

    public record Graph(String name, List<Node> nodes, Map<String, TensorData> initializers, List<ValueInfo> inputs, List<ValueInfo> outputs) {
    }

    private final Graph graph;
    private final Map<String, Long> opsets;

    private OnnxModel(Graph graph, Map<String, Long> opsets) {
        this.graph = graph;
        this.opsets = opsets;
    }

    /** A model from a graph built in memory (by jitNN's other importers), with its operator-set versions. */
    public static OnnxModel of(Graph graph, Map<String, Long> opsets) {
        return new OnnxModel(graph, opsets);
    }

    public Graph graph() {
        return graph;
    }

    /** The operator-set version of {@code domain} ("" is the default ONNX domain). */
    public long opset(String domain) {
        return opsets.getOrDefault(domain, 0L);
    }

    public static OnnxModel load(Path file) throws IOException {
        ByteBuffer bytes = map(file);
        Path dir = file.toAbsolutePath().getParent();
        Map<String, ByteBuffer> external = new HashMap<>();
        Proto p = new Proto(bytes);
        Graph g = null;
        Map<String, Long> opsets = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 7 -> g = graph(p.message(), dir, external);
                case 8 -> {
                    Proto o = p.message();
                    String domain = "";
                    long version = 0;
                    while (o.next()) {
                        switch (o.field()) {
                            case 1 -> domain = o.string();
                            case 2 -> version = o.varint();
                            default -> o.skip();
                        }
                    }
                    opsets.put(domain.equals("ai.onnx") ? "" : domain, version);
                }
                default -> p.skip();
            }
        }
        if (g == null) {
            throw new IOException(file + " has no graph: not an ONNX model");
        }
        return new OnnxModel(g, opsets);
    }

    private static ByteBuffer map(Path file) throws IOException {
        try (RandomAccessFile f = new RandomAccessFile(file.toFile(), "r"); FileChannel ch = f.getChannel()) {
            return ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN);
        }
    }

    static Graph graph(Proto p, Path dir, Map<String, ByteBuffer> external) throws IOException {
        String name = "";
        List<Node> nodes = new ArrayList<>();
        Map<String, TensorData> inits = new LinkedHashMap<>();
        List<ValueInfo> inputs = new ArrayList<>();
        List<ValueInfo> outputs = new ArrayList<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> nodes.add(node(p.message(), dir, external));
                case 2 -> name = p.string();
                case 5 -> {
                    TensorData t = tensor(p.message(), dir, external);
                    inits.put(t.name(), t);
                }
                case 11 -> inputs.add(valueInfo(p.message()));
                case 12 -> outputs.add(valueInfo(p.message()));
                default -> p.skip();
            }
        }
        // inputs that are also initializers are constants, not graph inputs
        inputs.removeIf(v -> inits.containsKey(v.name()));
        return new Graph(name, nodes, inits, inputs, outputs);
    }

    static Node node(Proto p, Path dir, Map<String, ByteBuffer> external) throws IOException {
        List<String> in = new ArrayList<>();
        List<String> out = new ArrayList<>();
        String name = "";
        String op = "";
        String domain = "";
        Map<String, Attribute> attrs = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> in.add(p.string());
                case 2 -> out.add(p.string());
                case 3 -> name = p.string();
                case 4 -> op = p.string();
                case 5 -> {
                    Attribute a = attribute(p.message(), dir, external);
                    attrs.put(a.name(), a);
                }
                case 7 -> domain = p.string();
                default -> p.skip();
            }
        }
        return new Node(name, op, domain, in, out, attrs);
    }

    static Attribute attribute(Proto p, Path dir, Map<String, ByteBuffer> external) throws IOException {
        String name = "";
        int type = 0;
        float f = 0;
        long i = 0;
        String s = null;
        TensorData t = null;
        Graph g = null;
        List<Float> floats = new ArrayList<>();
        List<Long> ints = new ArrayList<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> name = p.string();
                case 2 -> f = p.float32();
                case 3 -> i = p.varint();
                case 4 -> s = p.string();
                case 5 -> t = tensor(p.message(), dir, external);
                case 6 -> g = graph(p.message(), dir, external);
                case 7 -> p.floats(floats::add);
                case 8 -> p.longs(ints::add);
                case 20 -> type = (int) p.varint();
                default -> p.skip();
            }
        }
        float[] fa = new float[floats.size()];
        for (int k = 0; k < fa.length; k++) {
            fa[k] = floats.get(k);
        }
        long[] ia = ints.stream().mapToLong(Long::longValue).toArray();
        return new Attribute(name, type, f, i, s, t, fa, ia, g);
    }

    static ValueInfo valueInfo(Proto p) {
        String name = "";
        int elem = 0;
        List<Long> dims = new ArrayList<>();
        List<String> names = new ArrayList<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> name = p.string();
                case 2 -> {
                    Proto type = p.message();
                    while (type.next()) {
                        if (type.field() == 1) {
                            Proto tt = type.message();
                            while (tt.next()) {
                                switch (tt.field()) {
                                    case 1 -> elem = (int) tt.varint();
                                    case 2 -> {
                                        Proto shape = tt.message();
                                        while (shape.next()) {
                                            if (shape.field() == 1) {
                                                Proto dim = shape.message();
                                                long v = -1;
                                                String param = null;
                                                while (dim.next()) {
                                                    if (dim.field() == 1) {
                                                        v = dim.varint();
                                                    } else if (dim.field() == 2) {
                                                        param = dim.string();
                                                    } else {
                                                        dim.skip();
                                                    }
                                                }
                                                dims.add(v);
                                                names.add(v < 0 ? param : null);
                                            } else {
                                                shape.skip();
                                            }
                                        }
                                    }
                                    default -> tt.skip();
                                }
                            }
                        } else {
                            type.skip();
                        }
                    }
                }
                default -> p.skip();
            }
        }
        return new ValueInfo(name, elem, dims.stream().mapToLong(Long::longValue).toArray(), names.toArray(new String[0]));
    }

    static TensorData tensor(Proto p, Path dir, Map<String, ByteBuffer> external) throws IOException {
        List<Long> dims = new ArrayList<>();
        int type = 0;
        String name = "";
        ByteBuffer raw = null;
        List<Float> floatData = new ArrayList<>();
        List<Long> intData = new ArrayList<>();
        List<Double> doubleData = new ArrayList<>();
        boolean int64 = false;
        Map<String, String> ext = new HashMap<>();
        boolean isExternal = false;
        while (p.next()) {
            switch (p.field()) {
                case 1 -> p.longs(dims::add);
                case 2 -> type = (int) p.varint();
                case 4 -> p.floats(floatData::add);
                case 5 -> p.longs(intData::add);
                case 7 -> {
                    p.longs(intData::add);
                    int64 = true;
                }
                case 8 -> name = p.string();
                case 9 -> raw = p.bytes();
                case 10 -> p.doubles(doubleData::add);
                case 13 -> {
                    Proto e = p.message();
                    String k = "";
                    String v = "";
                    while (e.next()) {
                        switch (e.field()) {
                            case 1 -> k = e.string();
                            case 2 -> v = e.string();
                            default -> e.skip();
                        }
                    }
                    ext.put(k, v);
                }
                case 14 -> isExternal = p.varint() == 1;
                default -> p.skip();
            }
        }
        long[] d = dims.stream().mapToLong(Long::longValue).toArray();
        if (isExternal) {
            ByteBuffer file = external.get(ext.get("location"));
            if (file == null) {
                Path f = dir.resolve(ext.get("location"));
                if (!Files.exists(f)) {
                    throw new IOException("external data " + f + " of tensor " + name + " not found");
                }
                file = map(f);
                external.put(ext.get("location"), file);
            }
            long offset = Long.parseLong(ext.getOrDefault("offset", "0"));
            long length = ext.containsKey("length") ? Long.parseLong(ext.get("length")) : file.capacity() - offset;
            raw = file.slice((int) offset, (int) length).order(ByteOrder.LITTLE_ENDIAN);
        } else if (raw == null) {
            raw = typedData(type, floatData, intData, doubleData, int64);
        }
        return new TensorData(name, d, type, raw);
    }

    /** Re-encodes the typed repeated fields as the little-endian bytes raw_data would hold. */
    private static ByteBuffer typedData(int type, List<Float> floats, List<Long> ints, List<Double> doubles, boolean int64) {
        ByteBuffer b;
        switch (type) {
            case FLOAT -> {
                b = ByteBuffer.allocate(4 * floats.size()).order(ByteOrder.LITTLE_ENDIAN);
                floats.forEach(b::putFloat);
            }
            case DOUBLE -> {
                b = ByteBuffer.allocate(8 * doubles.size()).order(ByteOrder.LITTLE_ENDIAN);
                doubles.forEach(b::putDouble);
            }
            case INT64 -> {
                b = ByteBuffer.allocate(8 * ints.size()).order(ByteOrder.LITTLE_ENDIAN);
                ints.forEach(b::putLong);
            }
            case INT32 -> {
                b = ByteBuffer.allocate(4 * ints.size()).order(ByteOrder.LITTLE_ENDIAN);
                ints.forEach(v -> b.putInt((int) (long) v));
            }
            case FLOAT16, BFLOAT16, INT16, UINT16 -> {
                b = ByteBuffer.allocate(2 * ints.size()).order(ByteOrder.LITTLE_ENDIAN);
                ints.forEach(v -> b.putShort((short) (long) v));
            }
            case INT8, UINT8, BOOL -> {
                b = ByteBuffer.allocate(ints.size()).order(ByteOrder.LITTLE_ENDIAN);
                ints.forEach(v -> b.put((byte) (long) v));
            }
            default -> b = ByteBuffer.allocate(0);
        }
        b.flip();
        return b.order(ByteOrder.LITTLE_ENDIAN);
    }
}
