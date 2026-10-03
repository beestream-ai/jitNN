package org.beehive.jitnn;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** The PyTorch golden cases written by scripts/pytorch/goldens.py (test resources /golden). */
public final class Golden {

    public record Arr(String dtype, int[] dims, float[] f, int[] i) {
        public Tensor tensor() {
            return "i32".equals(dtype) ? Tensor.of(i, dims) : Tensor.of(f, dims);
        }
    }

    public record Case(String name, String op, String[] attrs, Map<String, Arr> arrays) {
        public Arr get(String role) {
            return arrays.get(role);
        }

        public float fattr(int k) {
            return Float.parseFloat(attrs[k]);
        }

        public int iattr(int k) {
            return Integer.parseInt(attrs[k]);
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private static List<Case> cases;

    private Golden() {
    }

    public static synchronized List<Case> cases() {
        if (cases == null) {
            try {
                cases = load();
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }
        return cases;
    }

    private static List<Case> load() throws IOException {
        byte[] data;
        String manifest;
        try (InputStream d = Golden.class.getResourceAsStream("/golden/data.bin"); InputStream m = Golden.class.getResourceAsStream("/golden/manifest.txt")) {
            data = d.readAllBytes();
            manifest = new String(m.readAllBytes(), StandardCharsets.UTF_8);
        }
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        Map<String, Case> byName = new LinkedHashMap<>();
        for (String line : manifest.split("\n")) {
            String[] t = line.trim().split("\\s+");
            if (t.length < 2 || t[0].startsWith("#")) {
                continue;
            }
            if (t[0].equals("case")) {
                String[] attrs = new String[t.length - 3];
                System.arraycopy(t, 3, attrs, 0, attrs.length);
                byName.put(t[1], new Case(t[1], t[2], attrs, new LinkedHashMap<>()));
                continue;
            }
            int[] dims = new int[t.length - 4];
            int n = 1;
            for (int k = 0; k < dims.length; k++) {
                dims[k] = Integer.parseInt(t[4 + k]);
                n *= dims[k];
            }
            int offset = Integer.parseInt(t[3]);
            Arr a;
            if (t[2].equals("i32")) {
                int[] v = new int[n];
                for (int k = 0; k < n; k++) {
                    v[k] = buf.getInt(offset + 4 * k);
                }
                a = new Arr("i32", dims, null, v);
            } else {
                float[] v = new float[n];
                for (int k = 0; k < n; k++) {
                    v[k] = buf.getFloat(offset + 4 * k);
                }
                a = new Arr("f32", dims, v, null);
            }
            byName.get(t[0]).arrays().put(t[1], a);
        }
        return new ArrayList<>(byName.values());
    }

    /** Tensors an operator created besides its output (batch norm's running statistics). */
    public static final ThreadLocal<Map<String, Tensor>> EXTRAS = ThreadLocal.withInitial(LinkedHashMap::new);

    /** Applies a golden case's operator with jitNN. */
    public static Tensor apply(Case c, Tensor[] in) {
        return switch (c.op()) {
            case "add" -> in[0].add(in[1]);
            case "sub" -> in[0].sub(in[1]);
            case "mul" -> in[0].mul(in[1]);
            case "div" -> in[0].div(in[1]);
            case "maximum" -> in[0].maximum(in[1]);
            case "neg" -> in[0].neg();
            case "exp" -> in[0].exp();
            case "log" -> in[0].log();
            case "sqrt" -> in[0].sqrt();
            case "abs" -> in[0].abs();
            case "relu" -> in[0].relu();
            case "reciprocal" -> in[0].reciprocal();
            case "square" -> in[0].square();
            case "pow" -> in[0].pow(c.fattr(0));
            case "clamp" -> in[0].clamp(c.fattr(0), c.fattr(1));
            case "affine" -> in[0].mul(c.fattr(0)).add(c.fattr(1));
            case "sum" -> in[0].sum(c.iattr(0), c.iattr(1) != 0);
            case "mean" -> in[0].mean(c.iattr(0), c.iattr(1) != 0);
            case "max" -> in[0].max(c.iattr(0), c.iattr(1) != 0);
            case "min" -> in[0].min(c.iattr(0), c.iattr(1) != 0);
            case "sum_all" -> in[0].sum();
            case "mean_all" -> in[0].mean();
            case "argmax" -> in[0].argmax(c.iattr(0));
            case "matmul" -> in[0].matmul(in[1], c.iattr(0) != 0, c.iattr(1) != 0);
            case "permute" -> {
                int[] order = new int[c.attrs().length];
                for (int k = 0; k < order.length; k++) {
                    order[k] = c.iattr(k);
                }
                yield in[0].permute(order);
            }
            case "reshape" -> {
                int[] dims = new int[c.attrs().length];
                for (int k = 0; k < dims.length; k++) {
                    dims[k] = c.iattr(k);
                }
                yield in[0].reshape(dims);
            }
            case "softmax" -> in[0].softmax();
            case "log_softmax" -> in[0].logSoftmax();
            case "cross_entropy" -> F.crossEntropy(in[0], in[1]);
            case "linear" -> F.linear(in[0], in[1], in[2]);
            case "linear_relu" -> F.linearRelu(in[0], in[1], in[2]);
            case "linear_nobias" -> F.linear(in[0], in[1], null);
            case "conv2d" -> F.conv2d(in[0], in[1], in[2], c.iattr(0), c.iattr(1), c.iattr(2), c.iattr(3));
            case "conv2d_nobias" -> F.conv2d(in[0], in[1], null, c.iattr(0), c.iattr(1), c.iattr(2), c.iattr(3));
            case "batch_norm" -> {
                int ch = in[0].dim(1);
                Tensor rm = Tensor.persistent(Tensor.zeros(ch));
                Tensor rv = Tensor.persistent(Tensor.ones(ch));
                EXTRAS.get().put("rm", rm);
                EXTRAS.get().put("rv", rv);
                yield F.batchNorm(in[0], rm, rv, in[1], in[2], true, 0.1f, 1e-5f);
            }
            case "max_pool2d" -> F.maxPool2d(in[0], c.iattr(0), c.iattr(1), c.iattr(2));
            case "avg_pool2d" -> F.avgPool2d(in[0], c.iattr(0), c.iattr(1), c.iattr(2));
            case "global_avg_pool" -> F.globalAvgPool2d(in[0]);
            case "relu6" -> in[0].relu6();
            case "tanh" -> in[0].tanh();
            case "sigmoid" -> in[0].sigmoid();
            case "gelu" -> in[0].gelu();
            case "gelu_tanh" -> in[0].geluTanh();
            case "silu" -> in[0].silu();
            case "swiglu" -> F.swiglu(in[0], in[1]);
            case "embedding" -> F.embedding(in[0], in[1], c.iattr(0));
            case "narrow" -> in[0].narrow(c.iattr(0), c.iattr(1), c.iattr(2));
            case "concat" -> Tensor.cat(c.iattr(0), in);
            case "stack" -> Tensor.stack(c.iattr(0), in);
            default -> throw new IllegalArgumentException("unknown golden op " + c.op());
        };
    }

    /** Tolerance class: exact-ish element-wise ops vs accumulating ones (reductions, GEMM, softmax). */
    public static double[] tolerance(Case c, boolean gradient) {
        boolean accumulates = switch (c.op()) {
            case "sum", "mean", "sum_all", "mean_all", "matmul", "softmax", "log_softmax", "cross_entropy", "linear", "linear_relu", "linear_nobias", "conv2d", "conv2d_nobias", "batch_norm", "global_avg_pool", "avg_pool2d", "embedding" -> true;
            default -> false;
        };
        double rtol = accumulates ? 1e-4 : 2e-6;
        double atol = accumulates ? 1e-4 : 1e-6;
        if (c.op().equals("pow") || c.op().equals("exp")) {
            rtol = 1e-5;
        }
        return gradient ? new double[] { rtol * 10, atol * 10 } : new double[] { rtol, atol };
    }
}
