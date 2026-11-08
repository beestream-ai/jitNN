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
package ai.beestream.jitnn.tf;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitnn.onnx.Proto;

/**
 * TensorFlow's graph messages (GraphDef, NodeDef, AttrValue, TensorProto, FunctionDef, SavedModel),
 * decoded with jitnn-onnx's protobuf reader: only the fields the importer needs.
 */
final class TfGraph {

    // DataType
    static final int DT_FLOAT = 1;
    static final int DT_DOUBLE = 2;
    static final int DT_INT32 = 3;
    static final int DT_UINT8 = 4;
    static final int DT_INT16 = 5;
    static final int DT_INT8 = 6;
    static final int DT_STRING = 7;
    static final int DT_INT64 = 9;
    static final int DT_BOOL = 10;
    static final int DT_BFLOAT16 = 14;
    static final int DT_HALF = 19;
    static final int DT_RESOURCE = 20;

    private TfGraph() {
    }

    /** A tensor value: element type, dims and little-endian raw bytes in the type's natural width. */
    record Tensor(int dtype, long[] dims, ByteBuffer raw, List<String> strings) {

        Tensor(int dtype, long[] dims, ByteBuffer raw) {
            this(dtype, dims, raw, List.of());
        }

        long numel() {
            long n = 1;
            for (long d : dims) {
                n *= d;
            }
            return n;
        }

        float[] floats() {
            int n = Math.toIntExact(numel());
            float[] out = new float[n];
            ByteBuffer b = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                out[i] = switch (dtype) {
                    case DT_FLOAT -> b.getFloat(4 * i);
                    case DT_DOUBLE -> (float) b.getDouble(8 * i);
                    case DT_HALF -> Float.float16ToFloat(b.getShort(2 * i));
                    case DT_BFLOAT16 -> Float.intBitsToFloat((b.getShort(2 * i) & 0xFFFF) << 16);
                    case DT_INT32 -> b.getInt(4 * i);
                    case DT_INT64 -> b.getLong(8 * i);
                    case DT_INT16 -> b.getShort(2 * i);
                    case DT_INT8 -> b.get(i);
                    case DT_UINT8, DT_BOOL -> b.get(i) & 0xFF;
                    default -> throw new UnsupportedOperationException("TensorFlow data type " + dtype);
                };
            }
            return out;
        }

        long[] longs() {
            int n = Math.toIntExact(numel());
            long[] out = new long[n];
            ByteBuffer b = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                out[i] = switch (dtype) {
                    case DT_INT32 -> b.getInt(4 * i);
                    case DT_INT64 -> b.getLong(8 * i);
                    case DT_INT16 -> b.getShort(2 * i);
                    case DT_INT8 -> b.get(i);
                    case DT_UINT8, DT_BOOL -> b.get(i) & 0xFF;
                    case DT_FLOAT -> (long) b.getFloat(4 * i);
                    case DT_DOUBLE -> (long) b.getDouble(8 * i);
                    default -> throw new UnsupportedOperationException("TensorFlow data type " + dtype + " as integers");
                };
            }
            return out;
        }

        static int width(int dtype) {
            return switch (dtype) {
                case DT_FLOAT, DT_INT32 -> 4;
                case DT_DOUBLE, DT_INT64 -> 8;
                case DT_HALF, DT_BFLOAT16, DT_INT16 -> 2;
                case DT_INT8, DT_UINT8, DT_BOOL -> 1;
                default -> 0;
            };
        }
    }

    /** An attribute value; which fields are set depends on its kind. */
    record Attr(String s, long i, float f, boolean b, int type, long[] shape, Tensor tensor, String func, List<Long> ints, List<Float> floats, List<Integer> types,
            List<String> strings, List<long[]> shapes) {

        static Attr empty() {
            return new Attr(null, 0, 0, false, 0, null, null, null, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        }
    }

    record Node(String name, String op, List<String> inputs, Map<String, Attr> attrs) {

        Attr attr(String key) {
            return attrs.get(key);
        }

        long i(String key, long dflt) {
            Attr a = attrs.get(key);
            return a == null ? dflt : a.i();
        }

        float f(String key, float dflt) {
            Attr a = attrs.get(key);
            return a == null ? dflt : a.f();
        }

        boolean b(String key, boolean dflt) {
            Attr a = attrs.get(key);
            return a == null ? dflt : a.b();
        }

        String s(String key, String dflt) {
            Attr a = attrs.get(key);
            return a == null || a.s() == null ? dflt : a.s();
        }

        int type(String key) {
            Attr a = attrs.get(key);
            return a == null ? 0 : a.type();
        }

        long[] ints(String key) {
            Attr a = attrs.get(key);
            return a == null ? null : a.ints().stream().mapToLong(Long::longValue).toArray();
        }

        /** Data inputs only (control inputs, "^name", dropped). */
        List<String> data() {
            List<String> d = new ArrayList<>();
            for (String in : inputs) {
                if (!in.startsWith("^")) {
                    d.add(in);
                }
            }
            return d;
        }
    }

    record Function(String name, List<String> inputArgs, List<String> outputArgs, List<Node> nodes, Map<String, String> ret) {
    }

    record Graph(List<Node> nodes, Map<String, Function> functions) {
    }

    record Signature(Map<String, String> inputs, Map<String, String> outputs) {
    }

    record SavedModel(Graph graph, Map<String, Signature> signatures, String restoreOp, String filenameTensor) {
    }

    // ---------------------------------------------------------------- decoding

    static Graph graph(Proto p) {
        List<Node> nodes = new ArrayList<>();
        Map<String, Function> functions = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> nodes.add(node(p.message()));
                case 2 -> {
                    Proto lib = p.message();
                    while (lib.next()) {
                        if (lib.field() == 1) {
                            Function f = function(lib.message());
                            functions.put(f.name(), f);
                        } else {
                            lib.skip();
                        }
                    }
                }
                default -> p.skip();
            }
        }
        return new Graph(nodes, functions);
    }

    static Node node(Proto p) {
        String name = "";
        String op = "";
        List<String> inputs = new ArrayList<>();
        Map<String, Attr> attrs = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> name = p.string();
                case 2 -> op = p.string();
                case 3 -> inputs.add(p.string());
                case 5 -> mapEntry(p.message(), attrs);
                default -> p.skip();
            }
        }
        return new Node(name, op, inputs, attrs);
    }

    private static void mapEntry(Proto e, Map<String, Attr> into) {
        String key = "";
        Attr value = Attr.empty();
        while (e.next()) {
            switch (e.field()) {
                case 1 -> key = e.string();
                case 2 -> value = attr(e.message());
                default -> e.skip();
            }
        }
        into.put(key, value);
    }

    static Attr attr(Proto p) {
        String s = null;
        long i = 0;
        float f = 0;
        boolean b = false;
        int type = 0;
        long[] shape = null;
        Tensor tensor = null;
        String func = null;
        List<Long> ints = new ArrayList<>();
        List<Float> floats = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        List<String> strings = new ArrayList<>();
        List<long[]> shapes = new ArrayList<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> {
                    Proto l = p.message();
                    while (l.next()) {
                        switch (l.field()) {
                            case 2 -> strings.add(l.string());
                            case 3 -> l.longs(ints::add);
                            case 4 -> l.floats(floats::add);
                            case 6 -> l.longs(v -> types.add((int) v));
                            case 7 -> shapes.add(shape(l.message()));
                            default -> l.skip();
                        }
                    }
                }
                case 2 -> s = p.string();
                case 3 -> i = p.varint();
                case 4 -> f = p.float32();
                case 5 -> b = p.varint() != 0;
                case 6 -> type = (int) p.varint();
                case 7 -> shape = shape(p.message());
                case 8 -> tensor = tensor(p.message());
                case 10 -> {
                    Proto n = p.message();
                    while (n.next()) {
                        if (n.field() == 1) {
                            func = n.string();
                        } else {
                            n.skip();
                        }
                    }
                }
                default -> p.skip();
            }
        }
        return new Attr(s, i, f, b, type, shape, tensor, func, ints, floats, types, strings, shapes);
    }

    /** Dims of a TensorShapeProto (-1 for unknown sizes); null for an unknown rank. */
    static long[] shape(Proto p) {
        List<Long> dims = new ArrayList<>();
        boolean unknownRank = false;
        while (p.next()) {
            switch (p.field()) {
                case 2 -> {
                    Proto d = p.message();
                    long size = -1;
                    while (d.next()) {
                        if (d.field() == 1) {
                            size = d.varint();
                        } else {
                            d.skip();
                        }
                    }
                    dims.add(size);
                }
                case 3 -> unknownRank = p.varint() != 0;
                default -> p.skip();
            }
        }
        return unknownRank ? null : dims.stream().mapToLong(Long::longValue).toArray();
    }

    static Tensor tensor(Proto p) {
        int dtype = 0;
        long[] dims = new long[0];
        ByteBuffer content = null;
        List<Float> fv = new ArrayList<>();
        List<Double> dv = new ArrayList<>();
        List<Long> iv = new ArrayList<>();
        List<String> sv = new ArrayList<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> dtype = (int) p.varint();
                case 8 -> sv.add(p.string());
                case 2 -> dims = shape(p.message());
                case 4 -> content = p.bytes();
                case 5 -> p.floats(fv::add);
                case 6 -> p.doubles(dv::add);
                case 7, 10, 11, 13 -> p.longs(iv::add);
                default -> p.skip();
            }
        }
        if (dims == null) {
            dims = new long[0];
        }
        if (dtype == DT_STRING) {
            return new Tensor(dtype, dims, ByteBuffer.allocate(0), sv);
        }
        int w = Tensor.width(dtype);
        long n = 1;
        for (long d : dims) {
            n *= d;
        }
        if (content != null && content.remaining() >= n * w) {
            return new Tensor(dtype, dims, content);
        }
        // typed value fields: fewer values than elements repeat the last one (a splat)
        ByteBuffer raw = ByteBuffer.allocate((int) (n * w)).order(ByteOrder.LITTLE_ENDIAN);
        int count = dtype == DT_FLOAT ? fv.size() : dtype == DT_DOUBLE ? dv.size() : iv.size();
        for (int k = 0; k < n; k++) {
            int src = Math.min(k, count - 1);
            if (count == 0) {
                break;
            }
            switch (dtype) {
                case DT_FLOAT -> raw.putFloat(fv.get(src));
                case DT_DOUBLE -> raw.putDouble(dv.get(src));
                case DT_INT64 -> raw.putLong(iv.get(src));
                case DT_INT32 -> raw.putInt(iv.get(src).intValue());
                case DT_HALF, DT_BFLOAT16, DT_INT16 -> raw.putShort(iv.get(src).shortValue());
                case DT_INT8, DT_UINT8, DT_BOOL -> raw.put(iv.get(src).byteValue());
                default -> throw new UnsupportedOperationException("TensorFlow constant of data type " + dtype);
            }
        }
        raw.rewind();
        return new Tensor(dtype, dims, raw);
    }

    static Function function(Proto p) {
        String name = "";
        List<String> in = new ArrayList<>();
        List<String> out = new ArrayList<>();
        List<Node> nodes = new ArrayList<>();
        Map<String, String> ret = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> {
                    Proto sig = p.message();
                    while (sig.next()) {
                        switch (sig.field()) {
                            case 1 -> name = sig.string();
                            case 2 -> in.add(argName(sig.message()));
                            case 3 -> out.add(argName(sig.message()));
                            default -> sig.skip();
                        }
                    }
                }
                case 3 -> nodes.add(node(p.message()));
                case 4 -> {
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
                    ret.put(k, v);
                }
                default -> p.skip();
            }
        }
        return new Function(name, in, out, nodes, ret);
    }

    private static String argName(Proto a) {
        String n = "";
        while (a.next()) {
            if (a.field() == 1) {
                n = a.string();
            } else {
                a.skip();
            }
        }
        return n;
    }

    static SavedModel savedModel(Proto p) {
        while (p.next()) {
            if (p.field() == 2) {
                return metaGraph(p.message());
            }
            p.skip();
        }
        throw new IllegalArgumentException("not a SavedModel: no meta graph");
    }

    private static SavedModel metaGraph(Proto p) {
        Graph graph = null;
        Map<String, Signature> signatures = new LinkedHashMap<>();
        String restore = null;
        String filename = null;
        while (p.next()) {
            switch (p.field()) {
                case 2 -> graph = graph(p.message());
                case 3 -> {
                    Proto s = p.message();
                    while (s.next()) {
                        switch (s.field()) {
                            case 1 -> filename = s.string();
                            case 3 -> restore = s.string();
                            default -> s.skip();
                        }
                    }
                }
                case 5 -> {
                    Proto e = p.message();
                    String key = "";
                    Signature sig = null;
                    while (e.next()) {
                        switch (e.field()) {
                            case 1 -> key = e.string();
                            case 2 -> sig = signature(e.message());
                            default -> e.skip();
                        }
                    }
                    signatures.put(key, sig);
                }
                default -> p.skip();
            }
        }
        return new SavedModel(graph, signatures, restore, filename);
    }

    private static Signature signature(Proto p) {
        Map<String, String> in = new LinkedHashMap<>();
        Map<String, String> out = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> tensorInfo(p.message(), in);
                case 2 -> tensorInfo(p.message(), out);
                default -> p.skip();
            }
        }
        return new Signature(in, out);
    }

    /** One entry of a map from a signature key to its TensorInfo: key to the graph tensor name. */
    private static void tensorInfo(Proto e, Map<String, String> into) {
        String key = "";
        String name = "";
        while (e.next()) {
            switch (e.field()) {
                case 1 -> key = e.string();
                case 2 -> {
                    Proto info = e.message();
                    while (info.next()) {
                        if (info.field() == 1) {
                            name = info.string();
                        } else {
                            info.skip();
                        }
                    }
                }
                default -> e.skip();
            }
        }
        into.put(key, name);
    }
}
