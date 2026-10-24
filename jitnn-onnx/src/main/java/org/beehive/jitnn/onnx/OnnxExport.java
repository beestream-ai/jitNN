package org.beehive.jitnn.onnx;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Storage;
import org.beehive.jitnn.kernels.ElementwiseKernels;

/**
 * Writes a jitNN forward pass as an ONNX model (opset 20), for ONNX Runtime and the other ONNX
 * tools. The forward pass is recorded once on example inputs, without running it: every recorded
 * operator becomes ONNX operators, parameters and other constants become initializers, and the
 * example inputs become the graph inputs. Shapes are those of the example inputs (no dynamic
 * axes). Weights are written in FP32. Operators without an ONNX mapping are reported by name.
 *
 * <pre>{@code
 * model.train(false);
 * OnnxExport.export(Path.of("model.onnx"), x -> model.forward(x[0]), new Tensor[] { example }, new String[] { "input" }, "output");
 * }</pre>
 */
public final class OnnxExport {

    private static final int OPSET = 20;
    private static final int FLOAT = OnnxModel.FLOAT;
    private static final int INT32 = OnnxModel.INT32;
    private static final int INT64 = OnnxModel.INT64;

    private record Val(String name, int[] shape, int type) {
    }

    private final Map<Storage, Val> values = new IdentityHashMap<>();
    private final Map<String, String> reshapes = new HashMap<>();
    private final List<ProtoWriter> nodes = new ArrayList<>();
    private final List<ProtoWriter> initializers = new ArrayList<>();
    private final Map<Storage, String> unavailable = new IdentityHashMap<>();
    private int counter;

    private OnnxExport() {
    }

    /** {@link #export(Path, Function, Tensor[], String[], String)} for a one-input model. */
    public static void export(Path file, Function<Tensor, Tensor> forward, Tensor example) throws IOException {
        export(file, x -> forward.apply(x[0]), new Tensor[] { example }, new String[] { "input" }, "output");
    }

    /**
     * Records {@code forward} on {@code inputs} (put the model in evaluation mode first) and writes
     * it to {@code file} with graph inputs {@code inputNames} and the output {@code outputName}.
     */
    public static void export(Path file, Function<Tensor[], Tensor> forward, Tensor[] inputs, String[] inputNames, String outputName) throws IOException {
        Tensor[] out = new Tensor[1];
        List<Node> recorded = Engine.capture(() -> out[0] = Grad.inference(() -> forward.apply(inputs)));
        OnnxExport x = new OnnxExport();
        ProtoWriter graph = new ProtoWriter();
        for (int i = 0; i < inputs.length; i++) {
            Tensor t = inputs[i];
            int type = elemType(t.dtype());
            x.values.put(t.storage(), new Val(inputNames[i], t.shape().dims(), type));
            graph.message(11, valueInfo(inputNames[i], type, t.shape().dims()));
        }
        for (Node n : recorded) {
            x.node(n);
        }
        String result = x.get(out[0]);
        x.add("Identity", List.of(result), List.of(outputName));
        for (ProtoWriter n : x.nodes) {
            graph.message(1, n);
        }
        graph.string(2, "jitnn");
        for (ProtoWriter i : x.initializers) {
            graph.message(5, i);
        }
        graph.message(12, valueInfo(outputName, elemType(out[0].dtype()), out[0].shape().dims()));
        ProtoWriter model = new ProtoWriter().varint(1, 9).string(2, "jitnn").string(3, "1").message(7, graph).message(8, new ProtoWriter().string(1, "").varint(2, OPSET));
        Files.write(file, model.bytes());
    }

    // ---------------------------------------------------------------- operators

    private void node(Node n) {
        int[] a = n.iattr;
        switch (n.op.name()) {
            case "linear" -> {
                List<String> in = new ArrayList<>(List.of(get(n.in(0), a[0], a[2]), get(n.in(1), a[1], a[2])));
                if (a[3] != 0) {
                    in.add(get(n.in(2), a[1]));
                }
                String y = one("Gemm", in, attrInt("transB", 1));
                y = switch (a[4]) {
                    case 0 -> y;
                    case 1 -> one("Relu", List.of(y));
                    case 2 -> one("Gelu", List.of(y));
                    case 3 -> one("Gelu", List.of(y), attrString("approximate", "tanh"));
                    default -> throw unsupported(n, "linear activation " + a[4]);
                };
                def(n.out(0), y, a[0], a[1]);
            }
            case "matmul" -> {
                int m = a[0], nn = a[1], k = a[2];
                boolean batched = a.length > 5;
                int batch = batched ? a[5] : 1;
                String x = operand(n.in(0), batched && a[6] != 0, batch, a[3] != 0 ? k : m, a[3] != 0 ? m : k, a[3] != 0);
                String w = operand(n.in(1), batched && a[7] != 0, batch, a[4] != 0 ? nn : k, a[4] != 0 ? k : nn, a[4] != 0);
                String y = one("MatMul", List.of(x, w));
                if (batched) {
                    def(n.out(0), y, batch, m, nn);
                } else {
                    def(n.out(0), y, m, nn);
                }
            }
            case "binary" -> binary(n);
            case "unary" -> unary(n);
            case "reduce" -> {
                String x = get(n.in(0), a[1], a[2], a[3]);
                String op = a[4] != 0 ? "ReduceMean" : switch (a[0]) {
                    case 0 -> "ReduceSum";
                    case 1 -> "ReduceMax";
                    case 2 -> "ReduceMin";
                    case 3 -> "ReduceProd";
                    default -> throw unsupported(n, "reduction " + a[0]);
                };
                def(n.out(0), one(op, List.of(x, constI64(1)), attrInt("keepdims", 0)), a[1], a[3]);
            }
            case "argmax", "argmin" -> {
                String x = get(n.in(0), a[0], a[1], a[2]);
                String i = one(n.op.name().equals("argmax") ? "ArgMax" : "ArgMin", List.of(x), attrInt("axis", 1), attrInt("keepdims", 0));
                defTyped(n.out(0), one("Cast", List.of(i), attrInt("to", INT32)), INT32, a[0], a[2]);
            }
            case "softmax" -> def(n.out(0), one(a[2] != 0 ? "LogSoftmax" : "Softmax", List.of(get(n.in(0), a[0], a[1])), attrInt("axis", -1)), a[0], a[1]);
            case "attention_softmax" -> {
                int rows = a[0], s = a[1], t = a[3];
                String x = one("Mul", List.of(get(n.in(0), rows / t, t, s), constF(n.fattr[0])));
                if (a[2] != 0) {
                    x = one("Add", List.of(x, causalMask(t, s)));
                }
                def(n.out(0), one("Softmax", List.of(x), attrInt("axis", -1)), rows / t, t, s);
            }
            case "row_norm" -> {
                int rows = a[0], cols = a[1];
                String x = get(n.in(0), rows, cols);
                String w = get(n.in(1), cols);
                String y;
                if (a[2] == 0) {
                    List<String> in = new ArrayList<>(List.of(x, w));
                    if (a[3] != 0) {
                        in.add(get(n.in(2), cols));
                    }
                    y = one("LayerNormalization", in, attrInt("axis", -1), attrFloat("epsilon", n.fattr[0]));
                } else {
                    String ms = one("ReduceMean", List.of(one("Mul", List.of(x, x)), constI64(-1)), attrInt("keepdims", 1));
                    String r = one("Reciprocal", List.of(one("Sqrt", List.of(one("Add", List.of(ms, constF(n.fattr[0])))))));
                    y = one("Mul", List.of(one("Mul", List.of(x, r)), w));
                }
                def(n.out(0), y, rows, cols);
                unavailable(n, 1, 2);
            }
            case "embedding" -> {
                String idx = get(n.in(0));
                def(n.out(0), one("Gather", List.of(get(n.in(1), a[1], a[2]), idx), attrInt("axis", 0)), concat(n.in(0).shape().dims(), a[2]));
            }
            case "permute" -> def(n.out(0), one("Transpose", List.of(get(n.in(0))), attrInts("perm", a)), n.out(0).shape().dims());
            case "narrow" -> {
                int outer = a[0], size = a[1], len = a[2], inner = a[3], start = a[4];
                String x = get(n.in(0), outer, size, inner);
                def(n.out(0), one("Slice", List.of(x, constI64(start), constI64(start + len), constI64(1))), outer, len, inner);
            }
            case "concat" -> {
                int outer = a[0], total = a[1], inner = a[2];
                List<String> parts = new ArrayList<>();
                for (int p = 0; p < n.inputs.length; p++) {
                    parts.add(get(n.in(p), outer, a[4 + p], inner));
                }
                def(n.out(0), one("Concat", parts, attrInt("axis", 1)), outer, total, inner);
            }
            case "pool2d" -> {
                int k = a[6], s = a[7], p = a[8];
                String x = get(n.in(0), a[0], a[1], a[2], a[3]);
                String y = a[9] == 0 ? one("MaxPool", List.of(x), attrInts("kernel_shape", k, k), attrInts("strides", s, s), attrInts("pads", p, p, p, p))
                        : one("AveragePool", List.of(x), attrInts("kernel_shape", k, k), attrInts("strides", s, s), attrInts("pads", p, p, p, p), attrInt("count_include_pad", 1));
                def(n.out(0), y, a[0], a[1], a[4], a[5]);
                unavailable(n, 1);
            }
            case "batch_norm" -> {
                if (a[3] != 0) {
                    throw unsupported(n, "BatchNorm in training mode (call train(false) before exporting)");
                }
                int[] shape = n.in(0).shape().dims();
                String y = one("BatchNormalization", List.of(get(n.in(0)), get(n.in(1), a[1]), get(n.in(2), a[1]), get(n.in(3), a[1]), get(n.in(4), a[1])),
                        attrFloat("epsilon", n.fattr[0]));
                def(n.out(0), a[4] != 0 ? one("Relu", List.of(y)) : y, shape);
            }
            case "conv2d" -> {
                int k = a[7], s = a[8], p = a[9], d = a[10];
                List<String> in = new ArrayList<>(List.of(get(n.in(0), a[0], a[1], a[2], a[3]), get(n.in(1), a[4], a[1] / a[11], k, k)));
                if (a[12] != 0) {
                    in.add(get(n.in(2), a[4]));
                }
                String y = one("Conv", in, attrInts("kernel_shape", k, k), attrInts("strides", s, s), attrInts("pads", p, p, p, p), attrInts("dilations", d, d),
                        attrInt("group", a[11]));
                def(n.out(0), a[13] != 0 ? one("Relu", List.of(y)) : y, a[0], a[4], a[5], a[6]);
            }
            case "flash_attention" -> attention(n);
            case "rope" -> rope(n);
            case "cast" -> {
                DType to = DType.values()[a[1]];
                String x = get(n.in(0));
                if (to == DType.I32) {
                    defTyped(n.out(0), one("Cast", List.of(x), attrInt("to", INT32)), INT32, n.in(0).shape().dims());
                } else if (to == DType.F32 || to == DType.F16 || to == DType.BF16) {
                    // the exported model computes in FP32
                    def(n.out(0), elemOf(n.in(0)) == FLOAT ? one("Identity", List.of(x)) : one("Cast", List.of(x), attrInt("to", FLOAT)), n.in(0).shape().dims());
                } else {
                    throw unsupported(n, "cast to " + to);
                }
            }
            case "dropout" -> def(n.out(0), one("Identity", List.of(get(n.in(0)))), n.in(0).shape().dims());
            default -> throw unsupported(n, null);
        }
    }

    private void binary(Node n) {
        int code = n.iattr[0];
        String x = get(n.in(0));
        String y = get(n.in(1));
        String r;
        if (code == ElementwiseKernels.ADD_RELU) {
            r = one("Relu", List.of(one("Add", List.of(x, y))));
        } else if (code == ElementwiseKernels.GT || code == ElementwiseKernels.LT || code == ElementwiseKernels.GE || code == ElementwiseKernels.LE || code == ElementwiseKernels.EQ
                || code == ElementwiseKernels.NE) {
            String op = code == ElementwiseKernels.GT ? "Greater" : code == ElementwiseKernels.LT ? "Less" : code == ElementwiseKernels.GE ? "GreaterOrEqual"
                    : code == ElementwiseKernels.LE ? "LessOrEqual" : "Equal";
            String b = one(op, List.of(x, y));
            if (code == ElementwiseKernels.NE) {
                b = one("Not", List.of(b));
            }
            // jitNN's booleans are FP32 0 and 1
            r = one("Cast", List.of(b), attrInt("to", FLOAT));
        } else {
            String op = code == ElementwiseKernels.ADD ? "Add" : code == ElementwiseKernels.SUB ? "Sub" : code == ElementwiseKernels.MUL ? "Mul"
                    : code == ElementwiseKernels.DIV ? "Div" : code == ElementwiseKernels.MAX || code == ElementwiseKernels.MAXIMUM ? "Max"
                            : code == ElementwiseKernels.MIN || code == ElementwiseKernels.MINIMUM ? "Min" : code == ElementwiseKernels.POW ? "Pow"
                                    : code == ElementwiseKernels.PRELU ? "PRelu" : null;
            if (op == null) {
                throw unsupported(n, "binary operator " + code);
            }
            r = one(op, List.of(x, y));
        }
        def(n.out(0), r, n.out(0).shape().dims());
    }

    private void unary(Node n) {
        int code = n.iattr[0];
        float alpha = n.fattr[0];
        float beta = n.fattr[1];
        String x = get(n.in(0));
        String r = switch (code) {
            case ElementwiseKernels.NEG -> one("Neg", List.of(x));
            case ElementwiseKernels.EXP -> one("Exp", List.of(x));
            case ElementwiseKernels.LOG -> one("Log", List.of(x));
            case ElementwiseKernels.SQRT -> one("Sqrt", List.of(x));
            case ElementwiseKernels.ABS -> one("Abs", List.of(x));
            case ElementwiseKernels.RELU -> one("Relu", List.of(x));
            case ElementwiseKernels.AFFINE -> one("Add", List.of(one("Mul", List.of(x, constF(alpha))), constF(beta)));
            case ElementwiseKernels.POW_SCALAR -> one("Pow", List.of(x, constF(alpha)));
            case ElementwiseKernels.CLAMP -> one("Clip", List.of(x, constF(alpha), constF(beta)));
            case ElementwiseKernels.RECIPROCAL -> one("Reciprocal", List.of(x));
            case ElementwiseKernels.SQUARE -> one("Mul", List.of(x, x));
            case ElementwiseKernels.TANH -> one("Tanh", List.of(x));
            case ElementwiseKernels.SIGMOID -> one("Sigmoid", List.of(x));
            case ElementwiseKernels.GELU -> one("Gelu", List.of(x));
            case ElementwiseKernels.GELU_TANH -> one("Gelu", List.of(x), attrString("approximate", "tanh"));
            case ElementwiseKernels.SILU -> one("Mul", List.of(x, one("Sigmoid", List.of(x))));
            case ElementwiseKernels.LEAKY_RELU -> one("LeakyRelu", List.of(x), attrFloat("alpha", alpha));
            case ElementwiseKernels.SOFTPLUS -> one("Softplus", List.of(x));
            case ElementwiseKernels.ELU -> one("Elu", List.of(x), attrFloat("alpha", alpha));
            case ElementwiseKernels.SELU -> one("Selu", List.of(x));
            case ElementwiseKernels.CELU -> one("Celu", List.of(x), attrFloat("alpha", alpha));
            case ElementwiseKernels.MISH -> one("Mish", List.of(x));
            case ElementwiseKernels.SOFTSIGN -> one("Softsign", List.of(x));
            case ElementwiseKernels.HARD_SIGMOID -> one("HardSigmoid", List.of(x), attrFloat("alpha", 1.0f / 6.0f), attrFloat("beta", 0.5f));
            case ElementwiseKernels.HARD_SWISH -> one("HardSwish", List.of(x));
            case ElementwiseKernels.THRESHOLD -> one("ThresholdedRelu", List.of(x), attrFloat("alpha", alpha));
            case ElementwiseKernels.SIN -> one("Sin", List.of(x));
            case ElementwiseKernels.COS -> one("Cos", List.of(x));
            case ElementwiseKernels.TAN -> one("Tan", List.of(x));
            case ElementwiseKernels.ASIN -> one("Asin", List.of(x));
            case ElementwiseKernels.ACOS -> one("Acos", List.of(x));
            case ElementwiseKernels.ATAN -> one("Atan", List.of(x));
            case ElementwiseKernels.SINH -> one("Sinh", List.of(x));
            case ElementwiseKernels.COSH -> one("Cosh", List.of(x));
            case ElementwiseKernels.ASINH -> one("Asinh", List.of(x));
            case ElementwiseKernels.ACOSH -> one("Acosh", List.of(x));
            case ElementwiseKernels.ATANH -> one("Atanh", List.of(x));
            case ElementwiseKernels.ERF -> one("Erf", List.of(x));
            case ElementwiseKernels.ROUND -> one("Round", List.of(x));
            case ElementwiseKernels.FLOOR -> one("Floor", List.of(x));
            case ElementwiseKernels.CEIL -> one("Ceil", List.of(x));
            case ElementwiseKernels.SIGN -> one("Sign", List.of(x));
            case ElementwiseKernels.LOG1P -> one("Log", List.of(one("Add", List.of(x, constF(1)))));
            case ElementwiseKernels.EXPM1 -> one("Sub", List.of(one("Exp", List.of(x)), constF(1)));
            case ElementwiseKernels.FAKE_QUANT8 -> fakeQuant(x, alpha, beta);
            case ElementwiseKernels.FILL -> initializer(full(n.out(0).numel(), alpha), n.out(0).shape().dims());
            default -> throw unsupported(n, "unary operator " + code);
        };
        def(n.out(0), r, n.out(0).shape().dims());
    }

    /** FAKE_QUANT8 as QuantizeLinear and DequantizeLinear, UINT8 when the zero point fits it, else INT8. */
    private String fakeQuant(String x, float scale, float lo) {
        int zp = (int) -lo;
        boolean unsigned = zp >= 0 && zp <= 255;
        String z = unsigned ? constU8(zp) : constI8(-128 - (int) lo);
        String s = constF(scale);
        return one("DequantizeLinear", List.of(one("QuantizeLinear", List.of(x, s, z)), s, z));
    }

    private void attention(Node n) {
        int[] a = n.iattr;
        int z = a[0], t = a[1], s = a[2], d = a[3], heads = a[5], layout = a[10];
        int b = z / heads;
        String q;
        String k;
        String v;
        if (layout == 1) {
            String qkv = one("Transpose", List.of(get(n.in(0), b, t, 3, heads, d)), attrInts("perm", 2, 0, 3, 1, 4));
            q = one("Gather", List.of(qkv, constScalarI64(0)), attrInt("axis", 0));
            k = one("Gather", List.of(qkv, constScalarI64(1)), attrInt("axis", 0));
            v = one("Gather", List.of(qkv, constScalarI64(2)), attrInt("axis", 0));
        } else if (layout == 2) {
            q = one("Transpose", List.of(get(n.in(0), b, t, heads, d)), attrInts("perm", 0, 2, 1, 3));
            k = one("Transpose", List.of(get(n.in(1), b, s, heads, d)), attrInts("perm", 0, 2, 1, 3));
            v = one("Transpose", List.of(get(n.in(2), b, s, heads, d)), attrInts("perm", 0, 2, 1, 3));
        } else {
            q = get(n.in(0), b, heads, t, d);
            k = get(n.in(1), b, heads, s, d);
            v = get(n.in(2), b, heads, s, d);
        }
        String scores = one("Mul", List.of(one("MatMul", List.of(q, one("Transpose", List.of(k), attrInts("perm", 0, 1, 3, 2)))), constF(n.fattr[0])));
        int maskAt = layout == 1 ? 1 : 3;
        if (n.inputs.length > maskAt) {
            scores = one("Add", List.of(scores, get(n.in(maskAt))));
        }
        if (a[4] != 0) {
            scores = one("Add", List.of(scores, causalMask(t, s)));
        }
        String o = one("MatMul", List.of(one("Softmax", List.of(scores), attrInt("axis", -1)), v));
        if (layout == 1) {
            def(n.out(0), one("Transpose", List.of(o), attrInts("perm", 0, 2, 1, 3)), b, t, heads, d);
        } else if (layout == 2) {
            def(n.out(0), one("Transpose", List.of(o), attrInts("perm", 0, 2, 1, 3)), b, t, heads, d);
        } else {
            def(n.out(0), o, b, heads, t, d);
        }
        unavailable(n, 1);
    }

    /**
     * Rotary embedding of x viewed as [outer, T, group, D]: x * cos + rotate(x) * sin with the
     * tables' rows offset .. offset + T - 1 as constants, rotate-half or interleaved pairs.
     */
    private void rope(Node n) {
        int[] a = n.iattr;
        int seq = a[0], dim = a[1], offset = a[2], group = a[4], half = dim / 2;
        int outer = n.in(0).numel() / (seq * group * dim);
        float[] cosAll = n.in(1).toFloatArray();
        float[] sinAll = n.in(2).toFloatArray();
        float[] c = new float[seq * half];
        float[] s = new float[seq * half];
        for (int t = 0; t < seq; t++) {
            for (int p = 0; p < half; p++) {
                c[t * half + p] = cosAll[(t + offset) * half + p];
                s[t * half + p] = n.fattr[0] * sinAll[(t + offset) * half + p];
            }
        }
        String cos = initializer(c, seq, 1, half);
        String sin = initializer(s, seq, 1, half);
        String y;
        if (a[3] == 0) {
            String x = get(n.in(0), outer, seq, group, dim);
            String lo = one("Slice", List.of(x, constI64(0), constI64(half), constI64(3)));
            String hi = one("Slice", List.of(x, constI64(half), constI64(dim), constI64(3)));
            String ra = one("Sub", List.of(one("Mul", List.of(lo, cos)), one("Mul", List.of(hi, sin))));
            String rb = one("Add", List.of(one("Mul", List.of(lo, sin)), one("Mul", List.of(hi, cos))));
            y = one("Concat", List.of(ra, rb), attrInt("axis", 3));
            def(n.out(0), y, outer, seq, group, dim);
        } else {
            String x = get(n.in(0), outer, seq, group, half, 2);
            String lo = one("Gather", List.of(x, constScalarI64(0)), attrInt("axis", 4));
            String hi = one("Gather", List.of(x, constScalarI64(1)), attrInt("axis", 4));
            String ra = one("Sub", List.of(one("Mul", List.of(lo, cos)), one("Mul", List.of(hi, sin))));
            String rb = one("Add", List.of(one("Mul", List.of(lo, sin)), one("Mul", List.of(hi, cos))));
            String four = constI64(4);
            y = one("Concat", List.of(one("Unsqueeze", List.of(ra, four)), one("Unsqueeze", List.of(rb, four))), attrInt("axis", 4));
            def(n.out(0), y, outer, seq, group, half, 2);
        }
    }

    /** [T, S]: 0 where query t may see key j (j <= t), -inf elsewhere. */
    private String causalMask(int t, int s) {
        float[] m = new float[t * s];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < s; j++) {
                m[i * s + j] = j <= i ? 0.0f : Float.NEGATIVE_INFINITY;
            }
        }
        return initializer(m, t, s);
    }

    /** A matrix operand of MatMul as [batch,] rows x cols, transposed back when {@code transposed}. */
    private String operand(Tensor t, boolean perBatch, int batch, int rows, int cols, boolean transposed) {
        String x = perBatch ? get(t, batch, rows, cols) : get(t, rows, cols);
        if (!transposed) {
            return x;
        }
        return one("Transpose", List.of(x), attrInts("perm", perBatch ? new int[] { 0, 2, 1 } : new int[] { 1, 0 }));
    }

    // ---------------------------------------------------------------- values

    /** The ONNX value of {@code t} in t's own shape. */
    private String get(Tensor t) {
        return get(t, t.shape().dims());
    }

    /** The ONNX value of {@code t} as {@code shape} (a Reshape when it was produced in another). */
    private String get(Tensor t, int... shape) {
        Storage s = t.storage();
        if (unavailable.containsKey(s)) {
            throw new UnsupportedOperationException("ONNX export: " + unavailable.get(s) + " is read, which has no ONNX counterpart");
        }
        Val v = values.get(s);
        if (v == null) {
            v = leaf(t);
            values.put(s, v);
        }
        if (Arrays.equals(v.shape, shape)) {
            return v.name;
        }
        String key = v.name + Arrays.toString(shape);
        String r = reshapes.get(key);
        if (r == null) {
            long[] dims = new long[shape.length];
            for (int i = 0; i < shape.length; i++) {
                dims[i] = shape[i];
            }
            r = one("Reshape", List.of(v.name, constI64(dims)));
            reshapes.put(key, r);
        }
        return r;
    }

    /** A tensor no recorded operator produces: a parameter or another constant, written as an initializer. */
    private Val leaf(Tensor t) {
        int[] dims = t.shape().dims();
        if (t.dtype() == DType.I32) {
            int[] v = t.toIntArray();
            ByteBuffer b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
            for (int x : v) {
                b.putInt(x);
            }
            return new Val(addInitializer(INT32, dims, b.array()), dims, INT32);
        }
        if (!t.dtype().isFloat() || t.dtype().isBlockQuantized() || t.dtype() == DType.FP8_E4M3 || t.dtype() == DType.FP8_E5M2) {
            throw new UnsupportedOperationException("ONNX export: " + t.dtype() + " constants are not supported");
        }
        return new Val(initializer(t.toFloatArray(), dims), dims, FLOAT);
    }

    private void def(Tensor t, String name, int... shape) {
        values.put(t.storage(), new Val(name, shape, FLOAT));
    }

    private void defTyped(Tensor t, String name, int type, int... shape) {
        values.put(t.storage(), new Val(name, shape, type));
    }

    private void unavailable(Node n, int... outputs) {
        for (int o : outputs) {
            if (o < n.outputs.length) {
                unavailable.put(n.out(o).storage(), n.op.name() + " output " + o);
            }
        }
    }

    private int elemOf(Tensor t) {
        Val v = values.get(t.storage());
        return v != null ? v.type : elemType(t.dtype());
    }

    private static int elemType(DType d) {
        return d == DType.I32 ? INT32 : FLOAT;
    }

    // ---------------------------------------------------------------- graph building

    private record Attr(ProtoWriter w) {
    }

    private String one(String op, List<String> in, Attr... attrs) {
        String out = op.toLowerCase() + "_" + counter++;
        add(op, in, List.of(out), attrs);
        return out;
    }

    private void add(String op, List<String> in, List<String> out, Attr... attrs) {
        ProtoWriter n = new ProtoWriter();
        for (String s : in) {
            n.string(1, s);
        }
        for (String s : out) {
            n.string(2, s);
        }
        n.string(3, out.get(0) + "/" + op);
        n.string(4, op);
        for (Attr a : attrs) {
            n.message(5, a.w);
        }
        nodes.add(n);
    }

    private static Attr attrInt(String name, long v) {
        return new Attr(new ProtoWriter().string(1, name).varint(3, v).varint(20, 2));
    }

    private static Attr attrFloat(String name, float v) {
        return new Attr(new ProtoWriter().string(1, name).fixed32(2, v).varint(20, 1));
    }

    private static Attr attrString(String name, String v) {
        return new Attr(new ProtoWriter().string(1, name).string(4, v).varint(20, 3));
    }

    private static Attr attrInts(String name, int... v) {
        ProtoWriter w = new ProtoWriter().string(1, name);
        for (int x : v) {
            w.varint(8, x);
        }
        return new Attr(w.varint(20, 7));
    }

    private String constF(float v) {
        return initializer(new float[] { v });
    }

    private String constI64(long... v) {
        ByteBuffer b = ByteBuffer.allocate(8 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (long x : v) {
            b.putLong(x);
        }
        return addInitializer(INT64, new int[] { v.length }, b.array());
    }

    private String constScalarI64(long v) {
        return addInitializer(INT64, new int[0], ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array());
    }

    private String constU8(int v) {
        return addInitializer(OnnxModel.UINT8, new int[0], new byte[] { (byte) v });
    }

    private String constI8(int v) {
        return addInitializer(OnnxModel.INT8, new int[0], new byte[] { (byte) v });
    }

    private String initializer(float[] v, int... dims) {
        ByteBuffer b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putFloat(x);
        }
        return addInitializer(FLOAT, dims, b.array());
    }

    private String addInitializer(int type, int[] dims, byte[] raw) {
        String name = "const_" + counter++;
        ProtoWriter t = new ProtoWriter();
        for (int d : dims) {
            t.varint(1, d);
        }
        t.varint(2, type).string(8, name).bytes(9, raw);
        initializers.add(t);
        return name;
    }

    private static ProtoWriter valueInfo(String name, int type, int[] dims) {
        ProtoWriter shape = new ProtoWriter();
        for (int d : dims) {
            shape.message(1, new ProtoWriter().varint(1, d));
        }
        ProtoWriter tensor = new ProtoWriter().varint(1, type).message(2, shape);
        return new ProtoWriter().string(1, name).message(2, new ProtoWriter().message(1, tensor));
    }

    private static float[] full(int n, float v) {
        float[] f = new float[n];
        Arrays.fill(f, v);
        return f;
    }

    private static int[] concat(int[] a, int b) {
        int[] r = Arrays.copyOf(a, a.length + 1);
        r[a.length] = b;
        return r;
    }

    private static UnsupportedOperationException unsupported(Node n, String what) {
        return new UnsupportedOperationException("ONNX export: jitNN operator " + n.op.name() + (what != null ? " (" + what + ")" : "") + " has no ONNX mapping");
    }
}
