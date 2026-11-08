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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.F;
import ai.beestream.jitnn.JitnnConfig;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.onnx.OnnxModel.Node;

/**
 * The ONNX operators jitNN runs, each mapped onto jitNN operations. Integer-valued shape
 * arithmetic (Shape, Gather on shapes, Concat of dims, Range, ...) is folded on the host when its
 * inputs are host values. Unsupported operators and attribute combinations fail at load with the
 * node's name.
 */
final class OnnxOps {

    interface Handler {
        List<Value> apply(Node n, List<Value> in);
    }

    static final Map<String, Handler> HANDLERS = new HashMap<>();

    static {
        OnnxQuant.register(HANDLERS);
    }

    /** Fused attention up to this many (batch x heads) rows, unfused beyond ({@code -Djitnn.onnx.fusedAttentionRows}). */
    static final int FUSED_ATTENTION_ROWS = JitnnConfig.current().get(JitnnConfig.ONNX_FUSED_ATTENTION_ROWS);

    private OnnxOps() {
    }

    static boolean supports(String op) {
        return HANDLERS.containsKey(op);
    }

    static List<Value> one(Value v) {
        return List.of(v);
    }

    static List<Value> one(Tensor t) {
        return List.of(Value.of(t));
    }

    /** Input k as an FP32 tensor (device INT32 tensors, such as integer graph inputs, are converted). */
    static Tensor t(List<Value> in, int k) {
        Tensor x = in.get(k).tensor(true);
        return x.dtype() == DType.I32 ? x.to(DType.F32) : x;
    }

    /** An integer value as an INT32 tensor (host integers uploaded; device FP32 truncated). */
    static Tensor int32(Value v) {
        Tensor t = v.tensor(false);
        return t.dtype() == DType.I32 ? t : t.trunc().to(DType.I32);
    }

    /** An integer bitwise operator: on the host for shape arithmetic, else on INT32 device tensors. */
    static Handler bitwise(java.util.function.BinaryOperator<Tensor> f, java.util.function.LongBinaryOperator host) {
        return (n, in) -> {
            if (allHost(in)) {
                return one(hostBinary(in.get(0), in.get(1), host));
            }
            return one(Value.of(f.apply(int32(in.get(0)), int32(in.get(1))), in.get(0).type));
        };
    }

    static boolean present(List<Value> in, int k) {
        return k < in.size() && in.get(k) != null;
    }

    static int axis(long a, int rank) {
        int ax = (int) (a < 0 ? a + rank : a);
        if (ax < 0 || ax >= rank) {
            throw new IllegalArgumentException("axis " + a + " outside a rank-" + rank + " tensor");
        }
        return ax;
    }

    static long[] dimsOf(Tensor t) {
        long[] d = new long[t.rank()];
        for (int i = 0; i < d.length; i++) {
            d[i] = t.dim(i);
        }
        return d;
    }

    static int[] toInts(long[] v) {
        int[] o = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            o[i] = Math.toIntExact(v[i]);
        }
        return o;
    }

    /** The output letters of an einsum equation (implicit form: the letters seen once, sorted). */
    static String einsumOutput(String eq) {
        int arrow = eq.indexOf("->");
        if (arrow >= 0) {
            return eq.substring(arrow + 2);
        }
        StringBuilder out = new StringBuilder();
        String letters = eq.replace(",", "");
        letters.chars().sorted().distinct().forEach(c -> {
            if (letters.indexOf(c) == letters.lastIndexOf(c)) {
                out.append((char) c);
            }
        });
        return out.toString();
    }

    private static Tensor sumOut(Tensor t, StringBuilder letters, String keep) {
        for (int i = letters.length() - 1; i >= 0; i--) {
            if (keep.indexOf(letters.charAt(i)) < 0) {
                t = t.sum(i, false);
                letters.deleteCharAt(i);
            }
        }
        return t;
    }

    private static Tensor arrange(Tensor t, String from, String to) {
        int[] order = new int[to.length()];
        boolean identity = true;
        for (int i = 0; i < order.length; i++) {
            order[i] = from.indexOf(to.charAt(i));
            identity &= order[i] == i;
        }
        return identity ? t : t.permute(order);
    }

    private static void checkDistinct(String operand, String eq) {
        if (operand.chars().distinct().count() != operand.length()) {
            throw new UnsupportedOperationException("einsum " + eq + ": repeated letters within an operand (diagonals) are not supported");
        }
    }

    /** One-operand einsum: sums the letters missing from the output, then permutes. */
    static Tensor einsum1(String eq, Tensor a) {
        String out = einsumOutput(eq);
        String in = eq.contains("->") ? eq.substring(0, eq.indexOf("->")) : eq;
        checkDistinct(in, eq);
        StringBuilder letters = new StringBuilder(in);
        Tensor t = sumOut(a, letters, out);
        return arrange(t, letters.toString(), out);
    }

    /**
     * Two-operand einsum as one batched GEMM: the letters of both operands and the output are the
     * batch, those of both but not the output are contracted, the rest are each side's free axes.
     */
    static Tensor einsum2(Node n, String eq, Tensor a, Tensor b) {
        String out = einsumOutput(eq);
        String lhs = eq.contains("->") ? eq.substring(0, eq.indexOf("->")) : eq;
        String[] ops = lhs.split(",");
        checkDistinct(ops[0], eq);
        checkDistinct(ops[1], eq);
        if (ops[0].length() != a.rank() || ops[1].length() != b.rank()) {
            throw unsupported(n, "equation " + eq + " for shapes " + a.shape() + " and " + b.shape());
        }
        // letters only one operand has and the output lacks are summed away first
        StringBuilder la = new StringBuilder(ops[0]);
        StringBuilder lb = new StringBuilder(ops[1]);
        a = sumOut(a, la, out + ops[1]);
        b = sumOut(b, lb, out + ops[0]);
        String sa = la.toString();
        String sb = lb.toString();
        StringBuilder batch = new StringBuilder();
        StringBuilder contract = new StringBuilder();
        StringBuilder left = new StringBuilder();
        StringBuilder right = new StringBuilder();
        for (char c : sa.toCharArray()) {
            if (sb.indexOf(c) >= 0) {
                (out.indexOf(c) >= 0 ? batch : contract).append(c);
            } else {
                left.append(c);
            }
        }
        for (char c : sb.toCharArray()) {
            if (sa.indexOf(c) < 0) {
                right.append(c);
            }
        }
        int bt = 1;
        int l = 1;
        int k = 1;
        int r = 1;
        java.util.List<Integer> outDims = new java.util.ArrayList<>();
        for (char c : batch.toString().toCharArray()) {
            int d = a.dim(sa.indexOf(c));
            if (d != b.dim(sb.indexOf(c))) {
                throw unsupported(n, "broadcasting batch letter '" + c + "' in " + eq);
            }
            bt *= d;
            outDims.add(d);
        }
        for (char c : left.toString().toCharArray()) {
            l *= a.dim(sa.indexOf(c));
            outDims.add(a.dim(sa.indexOf(c)));
        }
        for (char c : contract.toString().toCharArray()) {
            k *= a.dim(sa.indexOf(c));
        }
        for (char c : right.toString().toCharArray()) {
            r *= b.dim(sb.indexOf(c));
            outDims.add(b.dim(sb.indexOf(c)));
        }
        Tensor a3 = arrange(a, sa, batch.toString() + left + contract).reshape(bt, l, k);
        Tensor b3 = arrange(b, sb, batch.toString() + contract + right).reshape(bt, k, r);
        Tensor c = bt == 1 ? a3.reshape(l, k).matmul(b3.reshape(k, r)) : a3.matmul(b3);
        Tensor y = c.reshape(outDims.stream().mapToInt(Integer::intValue).toArray());
        return arrange(y, batch.toString() + left + right, out);
    }

    static UnsupportedOperationException unsupported(Node n, String what) {
        return new UnsupportedOperationException("ONNX " + n.opType() + " node '" + n.name() + "': " + what + " is not supported");
    }

    // ---------------------------------------------------------------- host integer arithmetic

    static long numel(long[] dims) {
        long n = 1;
        for (long d : dims) {
            n *= d;
        }
        return n;
    }

    /** Element-wise op on host integers with NumPy broadcasting. */
    static Value hostBinary(Value a, Value b, java.util.function.LongBinaryOperator op) {
        int rank = Math.max(a.rank(), b.rank());
        long[] dims = new long[rank];
        for (int i = 0; i < rank; i++) {
            long da = i - (rank - a.rank()) >= 0 ? a.dims[i - (rank - a.rank())] : 1;
            long db = i - (rank - b.rank()) >= 0 ? b.dims[i - (rank - b.rank())] : 1;
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException("cannot broadcast " + Arrays.toString(a.dims) + " and " + Arrays.toString(b.dims));
            }
            dims[i] = Math.max(da, db);
        }
        long[] out = new long[(int) numel(dims)];
        int[] idx = new int[rank];
        for (int k = 0; k < out.length; k++) {
            int rem = k;
            for (int i = rank - 1; i >= 0; i--) {
                idx[i] = (int) (rem % dims[i]);
                rem /= dims[i];
            }
            out[k] = op.applyAsLong(a.ints[flat(a.dims, idx, rank)], b.ints[flat(b.dims, idx, rank)]);
        }
        return Value.host(out, dims);
    }

    private static int flat(long[] dims, int[] idx, int rank) {
        int off = rank - dims.length;
        int f = 0;
        for (int i = 0; i < dims.length; i++) {
            f = (int) (f * dims[i] + (dims[i] == 1 ? 0 : idx[i + off]));
        }
        return f;
    }

    static boolean allHost(List<Value> in) {
        for (Value v : in) {
            if (v != null && !v.isHost()) {
                return false;
            }
        }
        return true;
    }

    // ---------------------------------------------------------------- shapes

    static long[] reshapeTarget(long[] in, long[] shape, boolean allowZero) {
        long[] out = shape.clone();
        int infer = -1;
        long known = 1;
        for (int i = 0; i < out.length; i++) {
            if (out[i] == 0 && !allowZero) {
                out[i] = in[i];
            }
            if (out[i] == -1) {
                infer = i;
            } else {
                known *= out[i];
            }
        }
        if (infer >= 0) {
            out[infer] = numel(in) / Math.max(known, 1);
        }
        return out;
    }

    static Value reshape(Value x, long[] shape) {
        if (x.isHost()) {
            return Value.host(x.ints, shape);
        }
        return Value.of(x.tensor.reshape(toInts(shape)), x.type);
    }

    static long[] axesFrom(Node n, List<Value> in, int inputIndex) {
        if (present(in, inputIndex)) {
            return in.get(inputIndex).longs();
        }
        return n.attrInts("axes", null);
    }

    static Value unsqueeze(Value x, long[] axes) {
        int rank = x.rank() + axes.length;
        boolean[] added = new boolean[rank];
        for (long a : axes) {
            added[axis(a, rank)] = true;
        }
        long[] dims = new long[rank];
        for (int i = 0, k = 0; i < rank; i++) {
            dims[i] = added[i] ? 1 : x.dims[k++];
        }
        return reshape(x, dims);
    }

    static Value squeeze(Value x, long[] axes) {
        List<Long> dims = new ArrayList<>();
        boolean[] drop = new boolean[x.rank()];
        if (axes == null) {
            for (int i = 0; i < x.rank(); i++) {
                drop[i] = x.dims[i] == 1;
            }
        } else {
            for (long a : axes) {
                drop[axis(a, x.rank())] = true;
            }
        }
        for (int i = 0; i < x.rank(); i++) {
            if (!drop[i]) {
                dims.add(x.dims[i]);
            }
        }
        return reshape(x, dims.stream().mapToLong(Long::longValue).toArray());
    }

    /** Host slice along several axes (step 1 or negative steps). */
    static Value hostSlice(Value x, long[] starts, long[] ends, long[] axes, long[] steps) {
        int rank = x.rank();
        long[] lo = new long[rank];
        long[] step = new long[rank];
        long[] len = x.dims.clone();
        Arrays.fill(step, 1);
        for (int k = 0; k < starts.length; k++) {
            int ax = axis(axes[k], rank);
            long d = x.dims[ax];
            long s = steps[k];
            long st = starts[k] < 0 ? starts[k] + d : starts[k];
            long en = ends[k] < 0 ? ends[k] + d : ends[k];
            if (s > 0) {
                st = Math.max(0, Math.min(st, d));
                en = Math.max(0, Math.min(en, d));
                len[ax] = Math.max(0, (en - st + s - 1) / s);
            } else {
                st = Math.max(-1, Math.min(st, d - 1));
                en = Math.max(-1, Math.min(en, d - 1));
                len[ax] = Math.max(0, (st - en - s - 1) / -s);
            }
            lo[ax] = st;
            step[ax] = s;
        }
        long[] out = new long[(int) numel(len)];
        int[] idx = new int[rank];
        for (int k = 0; k < out.length; k++) {
            int rem = k;
            for (int i = rank - 1; i >= 0; i--) {
                idx[i] = (int) (rem % len[i]);
                rem /= len[i];
            }
            long f = 0;
            for (int i = 0; i < rank; i++) {
                f = f * x.dims[i] + lo[i] + idx[i] * step[i];
            }
            out[k] = x.ints[(int) f];
        }
        return Value.host(out, len);
    }

    static Tensor deviceSlice(Node n, Tensor x, long[] starts, long[] ends, long[] axes, long[] steps) {
        if (x.dtype() == DType.I32) {
            // integer tensors are sliced in FP32 (exact up to 2^24) and converted back
            return deviceSlice(n, x.to(DType.F32), starts, ends, axes, steps).to(DType.I32);
        }
        Tensor y = x;
        for (int k = 0; k < starts.length; k++) {
            int ax = axis(axes[k], x.rank());
            long d = x.dim(ax);
            if (steps[k] != 1) {
                if (steps[k] == -1 && starts[k] >= d - 1 && ends[k] <= -d - 1) {
                    y = F.flip(y, ax);
                    continue;
                }
                // any other step: select the positions, ONNX's clamping rules
                long step = steps[k];
                long st = starts[k] < 0 ? starts[k] + d : starts[k];
                long en = ends[k] < 0 && ends[k] > Long.MIN_VALUE / 2 ? ends[k] + d : ends[k];
                java.util.List<Integer> pos = new java.util.ArrayList<>();
                if (step > 0) {
                    st = Math.max(0, Math.min(st, d));
                    en = Math.max(0, Math.min(en, d));
                    for (long i = st; i < en; i += step) {
                        pos.add((int) i);
                    }
                } else {
                    st = Math.max(-1, Math.min(st, d - 1));
                    en = Math.max(-1, Math.min(en, d - 1));
                    for (long i = st; i > en; i += step) {
                        pos.add((int) i);
                    }
                }
                int[] idx = pos.stream().mapToInt(Integer::intValue).toArray();
                y = y.indexSelect(ax, Tensor.of(idx, idx.length));
                continue;
            }
            long st = starts[k] < 0 ? starts[k] + d : starts[k];
            long en = ends[k] < 0 ? ends[k] + d : ends[k];
            st = Math.max(0, Math.min(st, d));
            en = Math.max(st, Math.min(en, d));
            if (st != 0 || en != d) {
                y = y.narrow(ax, (int) st, (int) (en - st));
            }
        }
        return y;
    }

    // ---------------------------------------------------------------- element-wise helpers

    static Handler unary(java.util.function.UnaryOperator<Tensor> f) {
        return (n, in) -> one(f.apply(t(in, 0)));
    }

    static Handler binary(java.util.function.BinaryOperator<Tensor> f, java.util.function.LongBinaryOperator host) {
        return (n, in) -> {
            if (host != null && allHost(in)) {
                return one(hostBinary(in.get(0), in.get(1), host));
            }
            return one(f.apply(t(in, 0), t(in, 1)));
        };
    }

    /** The scalar float value of a value with one element, when it is known on the host or a small constant. */
    static Tensor matmul(Tensor a, Tensor b) {
        boolean aVec = a.rank() == 1;
        boolean bVec = b.rank() == 1;
        Tensor a2 = aVec ? a.unsqueeze(0) : a;
        Tensor b2 = bVec ? b.unsqueeze(1) : b;
        if (a2.rank() > 2 && b2.rank() > 2 && a2.rank() == b2.rank()) {
            int r = a2.rank();
            int[] da = a2.shape().dims();
            int[] db = b2.shape().dims();
            int[] ea = da.clone();
            int[] eb = db.clone();
            for (int i = 0; i < r - 2; i++) {
                int m = Math.max(da[i], db[i]);
                ea[i] = m;
                eb[i] = m;
            }
            a2 = a2.expand(Shape.of(ea));
            b2 = b2.expand(Shape.of(eb));
        } else if (a2.rank() > 2 && b2.rank() > 2) {
            throw new UnsupportedOperationException("MatMul of ranks " + a2.rank() + " and " + b2.rank());
        }
        Tensor y = a2.matmul(b2);
        if (aVec) {
            y = y.squeeze(-2);
        }
        if (bVec) {
            y = y.squeeze(-1);
        }
        return y;
    }

    static Tensor softmax(Tensor x, int axis, boolean log) {
        int last = x.rank() - 1;
        Tensor y = axis == last ? x : x.transpose(axis, last);
        y = log ? y.logSoftmax() : y.softmax();
        return axis == last ? y : y.transpose(axis, last);
    }

    static Tensor reduce(Tensor x, long[] axes, boolean keep, String kind) {
        int rank = x.rank();
        int[] ax;
        if (axes == null || axes.length == 0) {
            ax = new int[rank];
            for (int i = 0; i < rank; i++) {
                ax[i] = i;
            }
        } else {
            ax = new int[axes.length];
            for (int i = 0; i < ax.length; i++) {
                ax[i] = axis(axes[i], rank);
            }
        }
        Arrays.sort(ax);
        Tensor y = x;
        for (int i = ax.length - 1; i >= 0; i--) {
            y = switch (kind) {
                case "mean" -> y.mean(ax[i], keep);
                case "sum" -> y.sum(ax[i], keep);
                case "max" -> y.max(ax[i], keep);
                case "min" -> y.min(ax[i], keep);
                case "sumsq" -> y.square().sum(ax[i], keep);
                case "prod" -> y.prod(ax[i], keep);
                default -> throw new IllegalArgumentException(kind);
            };
        }
        return y;
    }

    /** [begin, end] pads per spatial axis from ONNX pads [b1..bk, e1..ek], honouring auto_pad. */
    static long[] pads(Node n, int spatial, long[] in, long[] kernel, long[] strides, long[] dilations) {
        String auto = n.attrString("auto_pad", "NOTSET");
        long[] p = n.attrInts("pads", new long[2 * spatial]);
        if (auto.equals("SAME_UPPER") || auto.equals("SAME_LOWER")) {
            p = new long[2 * spatial];
            for (int i = 0; i < spatial; i++) {
                long out = (in[i] + strides[i] - 1) / strides[i];
                long total = Math.max(0, (out - 1) * strides[i] + (kernel[i] - 1) * dilations[i] + 1 - in[i]);
                long small = total / 2;
                p[i] = auto.equals("SAME_UPPER") ? small : total - small;
                p[i + spatial] = total - p[i];
            }
        } else if (auto.equals("VALID")) {
            p = new long[2 * spatial];
        }
        return p;
    }

    static int same(Node n, long[] v, String what) {
        for (long x : v) {
            if (x != v[0]) {
                throw unsupported(n, "different " + what + " per axis " + Arrays.toString(v));
            }
        }
        return (int) v[0];
    }

    /** x padded explicitly when the pads are asymmetric; returns [x, symmetric pad]. */
    static Object[] symmetric(Node n, Tensor x, long[] p, int spatial, float value) {
        boolean sym = true;
        for (int i = 0; i < spatial; i++) {
            sym &= p[i] == p[i + spatial] && p[i] == p[0];
        }
        if (sym) {
            return new Object[] { x, (int) p[0] };
        }
        int[] before = new int[spatial];
        int[] after = new int[spatial];
        for (int i = 0; i < spatial; i++) {
            before[i] = (int) p[i];
            after[i] = (int) p[i + spatial];
        }
        return new Object[] { F.pad(x, before, after, F.PadMode.CONSTANT, value), 0 };
    }

    static {
        // ---- element-wise arithmetic
        HANDLERS.put("Add", binary(Tensor::add, Long::sum));
        HANDLERS.put("Sub", binary(Tensor::sub, (a, b) -> a - b));
        HANDLERS.put("Mul", binary(Tensor::mul, (a, b) -> a * b));
        HANDLERS.put("Div", (n, in) -> {
            if (allHost(in)) {
                return one(hostBinary(in.get(0), in.get(1), (a, b) -> Math.floorDiv(a, b)));
            }
            return one(t(in, 0).div(t(in, 1)));
        });
        HANDLERS.put("Pow", (n, in) -> {
            float[] e = in.get(1).hostFloats();
            if (e != null && e.length == 1) {
                return one(e[0] == 2.0f ? t(in, 0).square() : e[0] == 1.0f ? t(in, 0) : e[0] == 0.5f ? t(in, 0).sqrt() : t(in, 0).pow(e[0]));
            }
            return one(t(in, 0).log().mul(t(in, 1)).exp());
        });
        HANDLERS.put("Max", (n, in) -> {
            Tensor y = t(in, 0);
            for (int k = 1; k < in.size(); k++) {
                y = y.maximum(t(in, k));
            }
            return one(y);
        });
        HANDLERS.put("Min", (n, in) -> {
            Tensor y = t(in, 0);
            for (int k = 1; k < in.size(); k++) {
                y = y.minimum(t(in, k));
            }
            return one(y);
        });
        HANDLERS.put("Sum", (n, in) -> {
            Tensor y = t(in, 0);
            for (int k = 1; k < in.size(); k++) {
                y = y.add(t(in, k));
            }
            return one(y);
        });
        HANDLERS.put("Mean", (n, in) -> {
            Tensor y = t(in, 0);
            for (int k = 1; k < in.size(); k++) {
                y = y.add(t(in, k));
            }
            return one(y.mul(1.0f / in.size()));
        });
        HANDLERS.put("Neg", (n, in) -> in.get(0).isHost() ? one(hostBinary(in.get(0), Value.scalar(-1), (a, b) -> a * b)) : one(t(in, 0).neg()));
        HANDLERS.put("Abs", unary(Tensor::abs));
        HANDLERS.put("Exp", unary(Tensor::exp));
        HANDLERS.put("Log", unary(Tensor::log));
        HANDLERS.put("Sqrt", unary(Tensor::sqrt));
        HANDLERS.put("Reciprocal", unary(Tensor::reciprocal));
        HANDLERS.put("Floor", unary(Tensor::floor));
        HANDLERS.put("Ceil", unary(Tensor::ceil));
        HANDLERS.put("Round", unary(Tensor::round));
        HANDLERS.put("Sign", unary(Tensor::sign));
        HANDLERS.put("Sin", unary(Tensor::sin));
        HANDLERS.put("Cos", unary(Tensor::cos));
        HANDLERS.put("Tan", unary(Tensor::tan));
        HANDLERS.put("Erf", unary(Tensor::erf));
        HANDLERS.put("Tanh", unary(Tensor::tanh));
        HANDLERS.put("Sigmoid", unary(Tensor::sigmoid));
        HANDLERS.put("Relu", unary(Tensor::relu));
        HANDLERS.put("Softplus", unary(Tensor::softplus));
        HANDLERS.put("Softsign", unary(Tensor::softsign));
        HANDLERS.put("Mish", unary(Tensor::mish));
        HANDLERS.put("HardSwish", unary(Tensor::hardSwish));
        HANDLERS.put("LeakyRelu", (n, in) -> one(t(in, 0).leakyRelu(n.attrFloat("alpha", 0.01f))));
        HANDLERS.put("Elu", (n, in) -> one(t(in, 0).elu(n.attrFloat("alpha", 1.0f))));
        HANDLERS.put("Selu", (n, in) -> one(t(in, 0).selu()));
        HANDLERS.put("Celu", (n, in) -> one(t(in, 0).celu(n.attrFloat("alpha", 1.0f))));
        HANDLERS.put("HardSigmoid", (n, in) -> one(t(in, 0).mul(n.attrFloat("alpha", 0.2f)).add(n.attrFloat("beta", 0.5f)).clamp(0.0f, 1.0f)));
        HANDLERS.put("PRelu", (n, in) -> {
            Tensor x = t(in, 0);
            Tensor slope = t(in, 1);
            return one(x.relu().sub(x.neg().relu().mul(slope)));
        });
        HANDLERS.put("Gelu", (n, in) -> one(n.attrString("approximate", "none").equals("tanh") ? t(in, 0).geluTanh() : t(in, 0).gelu()));
        HANDLERS.put("Clip", (n, in) -> {
            Tensor x = t(in, 0);
            float lo = present(in, 1) ? scalarOf(n, in.get(1)) : n.attrFloat("min", -Float.MAX_VALUE);
            float hi = present(in, 2) ? scalarOf(n, in.get(2)) : n.attrFloat("max", Float.MAX_VALUE);
            return one(x.clamp(lo, hi));
        });
        // ---- comparison and logic (booleans are FP32 0 / 1)
        HANDLERS.put("Equal", binary(Tensor::eq, (a, b) -> a == b ? 1 : 0));
        HANDLERS.put("Less", binary(Tensor::lt, (a, b) -> a < b ? 1 : 0));
        HANDLERS.put("LessOrEqual", binary(Tensor::le, (a, b) -> a <= b ? 1 : 0));
        HANDLERS.put("Greater", binary(Tensor::gt, (a, b) -> a > b ? 1 : 0));
        HANDLERS.put("GreaterOrEqual", binary(Tensor::ge, (a, b) -> a >= b ? 1 : 0));
        HANDLERS.put("And", binary(Tensor::logicalAnd, (a, b) -> a != 0 && b != 0 ? 1 : 0));
        HANDLERS.put("Or", binary(Tensor::logicalOr, (a, b) -> a != 0 || b != 0 ? 1 : 0));
        HANDLERS.put("Xor", binary(Tensor::logicalXor, (a, b) -> (a != 0) != (b != 0) ? 1 : 0));
        HANDLERS.put("Not", (n, in) -> one(t(in, 0).logicalNot()));
        HANDLERS.put("BitwiseAnd", bitwise(Tensor::bitwiseAnd, (a, b) -> a & b));
        HANDLERS.put("BitwiseOr", bitwise(Tensor::bitwiseOr, (a, b) -> a | b));
        HANDLERS.put("BitwiseXor", bitwise(Tensor::bitwiseXor, (a, b) -> a ^ b));
        HANDLERS.put("BitwiseNot", (n, in) -> {
            Value x = in.get(0);
            if (x.isHost()) {
                return one(hostBinary(x, Value.scalar(0), (a, b) -> ~a));
            }
            return one(Value.of(int32(x).bitwiseNot(), x.type));
        });
        HANDLERS.put("BitShift", (n, in) -> {
            boolean left = n.attrString("direction", "").equals("LEFT");
            return bitwise(left ? Tensor::shiftLeft : Tensor::shiftRight, left ? (a, b) -> a << b : (a, b) -> a >> b).apply(n, in);
        });
        HANDLERS.put("Mod", (n, in) -> {
            boolean fmod = n.attrInt("fmod", 0) != 0;
            if (allHost(in)) {
                return one(hostBinary(in.get(0), in.get(1), fmod ? (a, b) -> a % b : Math::floorMod));
            }
            boolean integral = in.get(0).type != OnnxModel.FLOAT && in.get(0).type != OnnxModel.FLOAT16 && in.get(0).type != OnnxModel.DOUBLE;
            if (integral) {
                Tensor a = int32(in.get(0));
                Tensor b = int32(in.get(1));
                return one(Value.of(fmod ? a.fmod(b) : a.remainder(b), in.get(0).type));
            }
            return one(fmod ? t(in, 0).fmod(t(in, 1)) : t(in, 0).remainder(t(in, 1)));
        });
        HANDLERS.put("DepthToSpace", (n, in) -> one(F.depthToSpace(t(in, 0), (int) n.attrInt("blocksize", 1), n.attrString("mode", "DCR"))));
        HANDLERS.put("SpaceToDepth", (n, in) -> one(F.spaceToDepth(t(in, 0), (int) n.attrInt("blocksize", 1))));
        HANDLERS.put("EyeLike", (n, in) -> {
            Value x = in.get(0);
            if (x.rank() != 2) {
                throw unsupported(n, "EyeLike of rank " + x.rank());
            }
            Tensor eye = Tensor.eye((int) x.dims[0], (int) x.dims[1], (int) n.attrInt("k", 0));
            int type = (int) n.attrInt("dtype", x.type);
            boolean integral = type != OnnxModel.FLOAT && type != OnnxModel.FLOAT16 && type != OnnxModel.DOUBLE;
            return one(integral ? Value.of(eye.to(DType.F32).trunc().to(DType.I32), type) : Value.of(eye));
        });
        HANDLERS.put("LpNormalization", (n, in) -> {
            Tensor x = t(in, 0);
            return one(F.lpNormalize(x, (int) n.attrInt("p", 2), axis(n.attrInt("axis", -1), x.rank()), 0.0f));
        });
        HANDLERS.put("Col2Im", (n, in) -> {
            Tensor cols = t(in, 0);
            int[] image = toInts(in.get(1).longs());
            int[] block = toInts(in.get(2).longs());
            int r = image.length;
            long[] pads = n.attrInts("pads", new long[2 * r]);
            int[] pad = new int[r];
            for (int a = 0; a < r; a++) {
                if (pads[a] != pads[a + r]) {
                    throw unsupported(n, "Col2Im with different begin and end padding");
                }
                pad[a] = (int) pads[a];
            }
            return one(F.fold(cols, image, block, toInts(n.attrInts("strides", ones(r))), pad, toInts(n.attrInts("dilations", ones(r)))));
        });
        HANDLERS.put("ScatterND", (n, in) -> {
            String reduction = n.attrString("reduction", "none");
            if (!(reduction.equals("none") || reduction.equals("add") || reduction.equals("mul"))) {
                throw unsupported(n, "ScatterND with reduction " + reduction);
            }
            return one(F.scatterND(t(in, 0), int32(in.get(1)), t(in, 2), reduction));
        });
        HANDLERS.put("IsNaN", unary(Tensor::isNaN));
        HANDLERS.put("IsInf", unary(Tensor::isInf));
        HANDLERS.put("Where", (n, in) -> {
            if (allHost(in)) {
                Value c = in.get(0);
                Value ab = hostBinary(in.get(1), in.get(2), (a, b) -> 0);
                Value pick = hostBinary(hostBinary(c, ab, (x, y) -> x), ab, (x, y) -> x);
                long[] out = new long[pick.ints.length];
                Value a = hostBinary(in.get(1), pick, (x, y) -> x);
                Value b = hostBinary(in.get(2), pick, (x, y) -> x);
                for (int i = 0; i < out.length; i++) {
                    out[i] = pick.ints[i] != 0 ? a.ints[i] : b.ints[i];
                }
                return one(Value.host(out, pick.dims));
            }
            return one(F.where(t(in, 0), t(in, 1), t(in, 2)));
        });
        // ---- matrix products
        HANDLERS.put("MatMul", (n, in) -> one(matmul(t(in, 0), t(in, 1))));
        HANDLERS.put("Einsum", (n, in) -> {
            String eq = n.attrString("equation", "").replace(" ", "");
            if (eq.contains("...")) {
                throw unsupported(n, "an ellipsis in equation " + eq);
            }
            if (in.size() == 1) {
                return one(Value.of(einsum1(eq, t(in, 0))));
            }
            if (in.size() == 2) {
                return one(Value.of(einsum2(n, eq, t(in, 0), t(in, 1))));
            }
            throw unsupported(n, in.size() + " operands");
        });
        HANDLERS.put(Rewrites.ATTENTION, (n, in) -> {
            Tensor q = t(in, 0);
            Tensor k = n.attrInt("form", 0) == 1 ? t(in, 1) : t(in, 1).transpose(-2, -1);
            Tensor v = t(in, 2);
            if (k.rank() != q.rank() || k.dim(-1) != q.dim(-1) || k.dim(-2) != v.dim(-2)) {
                throw new IllegalStateException("attention rewrite does not fit q " + q.shape() + ", k " + k.shape() + ", v " + v.shape()
                        + "; run with -Djitnn.onnx.rewrite=false and report the model");
            }
            float scale = n.attrFloat("scale", 1.0f);
            Tensor mask = present(in, 3) ? t(in, 3) : null;
            // the fused kernel wins while batch x heads leaves the GPU short of work; beyond that the
            // GEMMs with a fused scale-softmax between them are faster (measured on BERT-base, A100)
            int rows = 1;
            for (int i = 0; i < q.rank() - 2; i++) {
                rows *= q.dim(i);
            }
            if (rows <= FUSED_ATTENTION_ROWS) {
                return one(F.scaledDotProductAttention(q, k, v, mask, false, scale));
            }
            Tensor scores = q.matmul(k, false, true);
            Tensor p = mask == null ? F.attentionSoftmax(scores, scale, false)
                    : F.attentionSoftmax(scores.mul(scale).add(mask), 1.0f, false);
            return one(p.matmul(v));
        });
        HANDLERS.put(Rewrites.LINEAR, (n, in) -> one(F.linear(t(in, 0), t(in, 1), present(in, 2) ? t(in, 2) : null)));
        HANDLERS.put("Gemm", (n, in) -> {
            Tensor a = t(in, 0);
            Tensor b = t(in, 1);
            boolean ta = n.attrInt("transA", 0) != 0;
            boolean tb = n.attrInt("transB", 0) != 0;
            float alpha = n.attrFloat("alpha", 1.0f);
            float beta = n.attrFloat("beta", 1.0f);
            Tensor c = present(in, 2) ? t(in, 2) : null;
            Tensor y;
            if (!ta && tb && alpha == 1.0f && (c == null || (beta == 1.0f && c.rank() == 1))) {
                y = F.linear(a, b, c);
            } else {
                y = a.matmul(b, ta, tb);
                if (alpha != 1.0f) {
                    y = y.mul(alpha);
                }
                if (c != null) {
                    y = y.add(beta == 1.0f ? c : c.mul(beta));
                }
            }
            return one(y);
        });
        // ---- convolution and pooling
        HANDLERS.put("Conv", (n, in) -> {
            Tensor x = t(in, 0);
            Tensor w = t(in, 1);
            Tensor b = present(in, 2) ? t(in, 2) : null;
            int spatial = x.rank() - 2;
            if (spatial < 1 || spatial > 3) {
                throw unsupported(n, spatial + "-D convolution");
            }
            long[] kernel = Arrays.copyOfRange(dimsOf(w), 2, w.rank());
            long[] strides = n.attrInts("strides", ones(spatial));
            long[] dil = n.attrInts("dilations", ones(spatial));
            long[] p = pads(n, spatial, Arrays.copyOfRange(dimsOf(x), 2, x.rank()), kernel, strides, dil);
            return one(F.conv(x, w, b, toInts(strides), toInts(p), toInts(dil), (int) n.attrInt("group", 1)));
        });
        HANDLERS.put("ConvTranspose", (n, in) -> {
            Tensor x = t(in, 0);
            if (x.rank() != 4) {
                throw unsupported(n, "a non-2-D transposed convolution");
            }
            long[] p = n.attrInts("pads", new long[4]);
            if (p[0] != p[1] || p[0] != p[2] || p[0] != p[3]) {
                throw unsupported(n, "asymmetric pads");
            }
            long[] op = n.attrInts("output_padding", new long[2]);
            return one(F.convTranspose2d(x, t(in, 1), present(in, 2) ? t(in, 2) : null, same(n, n.attrInts("strides", ones(2)), "strides"), (int) p[0],
                    same(n, op, "output padding"), same(n, n.attrInts("dilations", ones(2)), "dilations"), (int) n.attrInt("group", 1)));
        });
        HANDLERS.put("MaxPool", (n, in) -> one(pool(n, t(in, 0), true)));
        HANDLERS.put("AveragePool", (n, in) -> one(pool(n, t(in, 0), false)));
        HANDLERS.put("LpPool", (n, in) -> one(pool(n, t(in, 0), F.PoolMode.LP, n.attrInt("p", 2))));
        HANDLERS.put("GlobalAveragePool", (n, in) -> {
            Tensor x = t(in, 0);
            for (int a = x.rank() - 1; a >= 2; a--) {
                x = x.mean(a, true);
            }
            return one(x);
        });
        HANDLERS.put("GlobalMaxPool", (n, in) -> {
            Tensor x = t(in, 0);
            for (int a = x.rank() - 1; a >= 2; a--) {
                x = x.max(a, true);
            }
            return one(x);
        });
        // ---- normalisation
        HANDLERS.put("BatchNormalization", (n, in) -> {
            Tensor x = t(in, 0);
            float eps = n.attrFloat("epsilon", 1e-5f);
            if (x.rank() == 4) {
                return one(F.batchNorm(x, t(in, 3), t(in, 4), t(in, 1), t(in, 2), false, 0.0f, eps));
            }
            int[] bd = new int[x.rank()];
            Arrays.fill(bd, 1);
            bd[1] = x.dim(1);
            Tensor scale = t(in, 1).div(t(in, 4).add(eps).sqrt());
            Tensor shift = t(in, 2).sub(t(in, 3).mul(scale));
            return one(x.mul(scale.reshape(bd)).add(shift.reshape(bd)));
        });
        HANDLERS.put("LayerNormalization", (n, in) -> {
            Tensor x = t(in, 0);
            int ax = axis(n.attrInt("axis", -1), x.rank());
            float eps = n.attrFloat("epsilon", 1e-5f);
            Tensor w = t(in, 1);
            Tensor b = present(in, 2) ? t(in, 2) : null;
            if (ax == x.rank() - 1) {
                return one(F.layerNorm(x, w, b, eps));
            }
            int[] dims = x.shape().dims();
            int inner = 1;
            for (int i = ax; i < dims.length; i++) {
                inner *= dims[i];
            }
            int[] flat = Arrays.copyOf(dims, ax + 1);
            flat[ax] = inner;
            Tensor y = F.layerNorm(x.reshape(flat), w.reshape(inner), b == null ? null : b.reshape(inner), eps);
            return one(y.reshape(dims));
        });
        HANDLERS.put("InstanceNormalization", (n, in) -> one(F.instanceNorm(t(in, 0), t(in, 1), t(in, 2), n.attrFloat("epsilon", 1e-5f))));
        HANDLERS.put("GroupNormalization", (n, in) -> {
            Tensor x = t(in, 0);
            int groups = (int) n.attrInt("num_groups", 1);
            Tensor scale = t(in, 1);
            Tensor bias = t(in, 2);
            if (scale.dim(0) != x.dim(1)) {
                // opset 18: one scale per group, repeated over the group's channels
                int per = x.dim(1) / groups;
                scale = scale.unsqueeze(1).expand(Shape.of(groups, per)).reshape(x.dim(1));
                bias = bias.unsqueeze(1).expand(Shape.of(groups, per)).reshape(x.dim(1));
            }
            return one(F.groupNorm(x, groups, scale, bias, n.attrFloat("epsilon", 1e-5f)));
        });
        HANDLERS.put("LRN", (n, in) -> one(F.localResponseNorm(t(in, 0), (int) n.attrInt("size", 5), n.attrFloat("alpha", 1e-4f), n.attrFloat("beta", 0.75f),
                n.attrFloat("bias", 1.0f))));
        HANDLERS.put("Softmax", (n, in) -> one(softmax(t(in, 0), axis(n.attrInt("axis", -1), t(in, 0).rank()), false)));
        HANDLERS.put("LogSoftmax", (n, in) -> one(softmax(t(in, 0), axis(n.attrInt("axis", -1), t(in, 0).rank()), true)));
        // ---- reductions (axes as an input since opset 18, an attribute before)
        for (String[] r : new String[][] { { "ReduceMean", "mean" }, { "ReduceSum", "sum" }, { "ReduceMax", "max" }, { "ReduceMin", "min" }, { "ReduceSumSquare", "sumsq" } }) {
            HANDLERS.put(r[0], (n, in) -> {
                long[] axes = axesFrom(n, in, 1);
                boolean keep = n.attrInt("keepdims", 1) != 0;
                if ((axes == null || axes.length == 0) && n.attrInt("noop_with_empty_axes", 0) != 0) {
                    return one(in.get(0));
                }
                return one(reduce(t(in, 0), axes, keep, r[1]));
            });
        }
        HANDLERS.put("ReduceProd", (n, in) -> one(reduce(t(in, 0), axesFrom(n, in, 1), n.attrInt("keepdims", 1) != 0, "prod")));
        HANDLERS.put("ReduceL2", (n, in) -> one(reduce(t(in, 0), axesFrom(n, in, 1), n.attrInt("keepdims", 1) != 0, "sumsq").sqrt()));
        HANDLERS.put("ArgMax", (n, in) -> {
            Tensor x = t(in, 0);
            int ax = axis(n.attrInt("axis", 0), x.rank());
            Tensor y = x.argmax(ax);
            return one(Value.of(n.attrInt("keepdims", 1) != 0 ? y.unsqueeze(ax) : y, OnnxModel.INT64));
        });
        HANDLERS.put("ArgMin", (n, in) -> {
            Tensor x = t(in, 0);
            int ax = axis(n.attrInt("axis", 0), x.rank());
            Tensor y = x.argmin(ax);
            return one(Value.of(n.attrInt("keepdims", 1) != 0 ? y.unsqueeze(ax) : y, OnnxModel.INT64));
        });
        HANDLERS.put("CumSum", (n, in) -> {
            if (n.attrInt("exclusive", 0) != 0 || n.attrInt("reverse", 0) != 0) {
                throw unsupported(n, "exclusive or reverse cumulative sums");
            }
            Tensor x = t(in, 0);
            return one(x.cumsum(axis(in.get(1).longs()[0], x.rank())));
        });
        HANDLERS.put("TopK", (n, in) -> {
            Tensor x = t(in, 0);
            int ax = axis(n.attrInt("axis", -1), x.rank());
            if (ax != x.rank() - 1 || n.attrInt("largest", 1) == 0) {
                throw unsupported(n, "top-k other than the largest along the last axis");
            }
            Tensor[] r = x.topk((int) in.get(1).longs()[0]);
            return List.of(Value.of(r[0]), Value.of(r[1], OnnxModel.INT64));
        });
        // ---- shape manipulation
        HANDLERS.put("Identity", (n, in) -> one(in.get(0)));
        HANDLERS.put("Dropout", (n, in) -> one(in.get(0)));
        HANDLERS.put("Shape", (n, in) -> {
            long[] d = in.get(0).dims;
            int start = (int) n.attrInt("start", 0);
            int end = (int) n.attrInt("end", d.length);
            start = start < 0 ? start + d.length : start;
            end = end < 0 ? end + d.length : Math.min(end, d.length);
            long[] s = Arrays.copyOfRange(d, Math.max(start, 0), Math.max(end, start));
            return one(Value.host(s, new long[] { s.length }));
        });
        HANDLERS.put("Size", (n, in) -> one(Value.scalar(in.get(0).numel())));
        HANDLERS.put("Reshape", (n, in) -> one(reshape(in.get(0), reshapeTarget(in.get(0).dims, in.get(1).longs(), n.attrInt("allowzero", 0) != 0))));
        HANDLERS.put("Flatten", (n, in) -> {
            Value x = in.get(0);
            int ax = (int) n.attrInt("axis", 1);
            ax = ax < 0 ? ax + x.rank() : ax;
            long outer = numel(Arrays.copyOfRange(x.dims, 0, ax));
            return one(reshape(x, new long[] { outer, x.numel() / Math.max(outer, 1) }));
        });
        HANDLERS.put("Squeeze", (n, in) -> one(squeeze(in.get(0), axesFrom(n, in, 1))));
        HANDLERS.put("Unsqueeze", (n, in) -> one(unsqueeze(in.get(0), axesFrom(n, in, 1))));
        HANDLERS.put("Transpose", (n, in) -> {
            Value x = in.get(0);
            // a second input carries a permutation computed in the graph (jitNN's TensorFlow importer)
            long[] perm = in.size() > 1 && present(in, 1) ? in.get(1).longs() : n.attrInts("perm", null);
            int[] p = new int[x.rank()];
            for (int i = 0; i < p.length; i++) {
                p[i] = perm == null ? p.length - 1 - i : (int) perm[i];
            }
            if (n.attrInt("jitnn_last2", 0) != 0) {
                // jitNN's TensorFlow importer: swap the last two axes of a tensor of any rank
                for (int i = 0; i < p.length; i++) {
                    p[i] = i;
                }
                p[p.length - 1] = p.length - 2;
                p[p.length - 2] = p.length - 1;
            }
            return one(Value.of(x.tensor(true).permute(p), x.type));
        });
        HANDLERS.put("Concat", (n, in) -> {
            int rank = in.get(0).rank();
            int ax = axis(n.attrInt("axis", 0), rank);
            if (allHost(in)) {
                long[] dims = in.get(0).dims.clone();
                dims[ax] = 0;
                for (Value v : in) {
                    dims[ax] += v.dims[ax];
                }
                long outer = numel(Arrays.copyOfRange(dims, 0, ax));
                long[] out = new long[(int) numel(dims)];
                int k = 0;
                for (long o = 0; o < outer; o++) {
                    for (Value v : in) {
                        long chunk = numel(Arrays.copyOfRange(v.dims, ax, v.rank()));
                        System.arraycopy(v.ints, (int) (o * chunk), out, k, (int) chunk);
                        k += (int) chunk;
                    }
                }
                return one(Value.host(out, dims));
            }
            List<Tensor> parts = new ArrayList<>();
            for (Value v : in) {
                if (v.numel() > 0) {
                    parts.add(v.tensor(true));
                }
            }
            return one(Tensor.cat(ax, parts.toArray(new Tensor[0])));
        });
        HANDLERS.put("Split", (n, in) -> {
            Tensor x = t(in, 0);
            int ax = axis(n.attrInt("axis", 0), x.rank());
            int outs = n.outputs().size();
            long[] sizes = present(in, 1) ? in.get(1).longs() : n.attrInts("split", null);
            if (sizes == null) {
                long each = (x.dim(ax) + outs - 1) / outs;
                sizes = new long[outs];
                for (int i = 0; i < outs; i++) {
                    sizes[i] = Math.min(each, x.dim(ax) - each * i);
                }
            }
            List<Value> out = new ArrayList<>();
            int at = 0;
            for (long s : sizes) {
                out.add(Value.of(x.narrow(ax, at, (int) s)));
                at += (int) s;
            }
            return out;
        });
        HANDLERS.put("Slice", (n, in) -> {
            Value x = in.get(0);
            long[] starts = present(in, 1) ? in.get(1).longs() : n.attrInts("starts", null);
            long[] ends = present(in, 2) ? in.get(2).longs() : n.attrInts("ends", null);
            long[] axes = present(in, 3) ? in.get(3).longs() : n.attrInts("axes", null);
            long[] steps = present(in, 4) ? in.get(4).longs() : null;
            if (axes == null) {
                axes = new long[starts.length];
                for (int i = 0; i < axes.length; i++) {
                    axes[i] = i;
                }
            }
            if (steps == null) {
                steps = ones(starts.length);
            }
            if (x.isHost()) {
                return one(hostSlice(x, starts, ends, axes, steps));
            }
            return one(Value.of(deviceSlice(n, x.tensor, starts, ends, axes, steps), x.type));
        });
        HANDLERS.put("Gather", (n, in) -> {
            Value data = in.get(0);
            Value idx = in.get(1);
            int ax = axis(n.attrInt("axis", 0), data.rank());
            if (data.isHost() && idx.isHost()) {
                long[] id = idx.ints.clone();
                long[] out = new long[id.length * (int) (data.numel() / data.dims[ax])];
                long outer = numel(Arrays.copyOfRange(data.dims, 0, ax));
                long inner = numel(Arrays.copyOfRange(data.dims, ax + 1, data.rank()));
                int k = 0;
                for (long o = 0; o < outer; o++) {
                    for (long i : id) {
                        long ii = i < 0 ? i + data.dims[ax] : i;
                        for (long q = 0; q < inner; q++) {
                            out[k++] = data.ints[(int) ((o * data.dims[ax] + ii) * inner + q)];
                        }
                    }
                }
                long[] dims = new long[data.rank() - 1 + idx.rank()];
                System.arraycopy(data.dims, 0, dims, 0, ax);
                System.arraycopy(idx.dims, 0, dims, ax, idx.rank());
                System.arraycopy(data.dims, ax + 1, dims, ax + idx.rank(), data.rank() - ax - 1);
                return one(Value.host(out, dims));
            }
            Tensor x = data.tensor(true);
            if (idx.isHost() && idx.numel() == 1 && idx.rank() == 0) {
                long i = idx.ints[0] < 0 ? idx.ints[0] + x.dim(ax) : idx.ints[0];
                return one(Value.of(x.narrow(ax, (int) i, 1).squeeze(ax), data.type));
            }
            Tensor ids = idx.tensor(false);
            if (ids.dtype() != DType.I32) {
                ids = ids.to(DType.I32);
            }
            if (ax == 0 && x.rank() == 2) {
                return one(Value.of(F.embedding(ids, x), data.type));
            }
            int[] dims = new int[x.rank() - 1 + ids.rank()];
            int[] xd = x.shape().dims();
            System.arraycopy(xd, 0, dims, 0, ax);
            System.arraycopy(ids.shape().dims(), 0, dims, ax, ids.rank());
            System.arraycopy(xd, ax + 1, dims, ax + ids.rank(), x.rank() - ax - 1);
            return one(Value.of(x.indexSelect(ax, ids.reshape(ids.numel())).reshape(dims), data.type));
        });
        HANDLERS.put("GatherND", (n, in) -> {
            Tensor x = t(in, 0);
            Value idx = in.get(1);
            if (n.attrInt("batch_dims", 0) != 0) {
                throw unsupported(n, "GatherND with batch_dims");
            }
            Tensor ids = int32(idx);
            if (idx.dims[idx.rank() - 1] != 1) {
                return one(F.gatherND(x, ids));
            }
            int[] id = ids.shape().dims();
            ids = ids.reshape(Arrays.copyOf(id, id.length - 1));
            if (x.rank() == 2) {
                return one(F.embedding(ids, x));
            }
            int[] dims = new int[ids.rank() + x.rank() - 1];
            System.arraycopy(ids.shape().dims(), 0, dims, 0, ids.rank());
            System.arraycopy(x.shape().dims(), 1, dims, ids.rank(), x.rank() - 1);
            return one(x.indexSelect(0, ids.reshape(ids.numel())).reshape(dims));
        });
        HANDLERS.put("Expand", (n, in) -> {
            Value x = in.get(0);
            long[] shape = in.get(1).longs();
            int rank = Math.max(shape.length, x.rank());
            long[] dims = new long[rank];
            for (int i = 0; i < rank; i++) {
                long a = i - (rank - x.rank()) >= 0 ? x.dims[i - (rank - x.rank())] : 1;
                long b = i - (rank - shape.length) >= 0 ? shape[i - (rank - shape.length)] : 1;
                dims[i] = Math.max(a, b);
            }
            if (x.isHost()) {
                return one(hostBinary(x, Value.host(new long[(int) numel(dims)], dims), (a, b) -> a));
            }
            Tensor xt = x.tensor;
            if (xt.rank() < rank) {
                int[] d = new int[rank];
                Arrays.fill(d, 1);
                System.arraycopy(xt.shape().dims(), 0, d, rank - xt.rank(), xt.rank());
                xt = xt.reshape(d);
            }
            return one(Value.of(xt.expand(Shape.of(toInts(dims))), x.type));
        });
        HANDLERS.put("Tile", (n, in) -> one(t(in, 0).tile(toInts(in.get(1).longs()))));
        HANDLERS.put("Cast", (n, in) -> {
            Value x = in.get(0);
            int to = (int) n.attrInt("to", OnnxModel.FLOAT);
            boolean integral = to == OnnxModel.INT64 || to == OnnxModel.INT32 || to == OnnxModel.INT8 || to == OnnxModel.UINT8 || to == OnnxModel.BOOL;
            if (x.isHost()) {
                if (to == OnnxModel.BOOL) {
                    long[] b = new long[x.ints.length];
                    for (int i = 0; i < b.length; i++) {
                        b[i] = x.ints[i] != 0 ? 1 : 0;
                    }
                    return one(Value.host(b, x.dims));
                }
                return integral ? one(x) : one(Value.of(x.tensor(true)));
            }
            Tensor xt = x.tensor;
            if (to == OnnxModel.BOOL) {
                return one(Value.of(xt.to(DType.F32).ne(Tensor.scalar(0.0f)), OnnxModel.BOOL));
            }
            if (integral) {
                return one(Value.of(xt.dtype() == DType.I32 ? xt : xt.trunc().to(DType.I32), OnnxModel.INT64));
            }
            return one(Value.of(xt.dtype() == DType.F32 ? xt : xt.to(DType.F32), OnnxModel.FLOAT));
        });
        HANDLERS.put("Range", (n, in) -> {
            if (allHost(in)) {
                long start = in.get(0).ints[0];
                long limit = in.get(1).ints[0];
                long delta = in.get(2).ints[0];
                int count = (int) Math.max(0, (limit - start + delta + (delta > 0 ? -1 : 1)) / delta);
                long[] out = new long[count];
                for (int i = 0; i < count; i++) {
                    out[i] = start + i * delta;
                }
                return one(Value.host(out, new long[] { count }));
            }
            float start = t(in, 0).toFloatArray()[0];
            float limit = t(in, 1).toFloatArray()[0];
            float delta = t(in, 2).toFloatArray()[0];
            return one(Tensor.arange(start, limit, delta));
        });
        HANDLERS.put("ConstantOfShape", (n, in) -> {
            long[] dims = in.get(0).longs();
            OnnxModel.Attribute v = n.attr("value");
            int type = v == null ? OnnxModel.FLOAT : v.t().dataType();
            if (type == OnnxModel.FLOAT || type == OnnxModel.FLOAT16 || type == OnnxModel.DOUBLE) {
                float f = v == null ? 0.0f : v.t().floats()[0];
                return one(Tensor.ones(toInts(dims)).mul(f));
            }
            long[] out = new long[(int) numel(dims)];
            Arrays.fill(out, v.t().longs()[0]);
            return one(Value.host(out, dims));
        });
        HANDLERS.put("Pad", (n, in) -> {
            Tensor x = t(in, 0);
            long[] p = present(in, 1) ? in.get(1).longs() : n.attrInts("pads", null);
            float value = present(in, 2) ? scalarOf(n, in.get(2)) : n.attrFloat("value", 0.0f);
            String mode = n.attrString("mode", "constant");
            int rank = x.rank();
            long[] axes = present(in, 3) ? in.get(3).longs() : null;
            long[] full = new long[2 * rank];
            for (int i = 0; i < p.length / 2; i++) {
                int ax = axes == null ? i : axis(axes[i], rank);
                full[ax] = p[i];
                full[ax + rank] = p[i + p.length / 2];
            }
            if (full[0] != 0 || full[rank] != 0 || full[1] != 0 || full[1 + rank] != 0 || rank < 3) {
                if (!mode.equals("constant")) {
                    throw unsupported(n, mode + " padding of the batch or channel axis");
                }
                // constant padding on any axis: concatenate blocks of the value
                Tensor y = x;
                for (int ax = 0; ax < rank; ax++) {
                    long b = full[ax];
                    long a = full[ax + rank];
                    if (b < 0 || a < 0) {
                        throw unsupported(n, "negative pads");
                    }
                    if (b == 0 && a == 0) {
                        continue;
                    }
                    java.util.List<Tensor> parts = new java.util.ArrayList<>();
                    int[] d = y.shape().dims().clone();
                    if (b > 0) {
                        d[ax] = (int) b;
                        parts.add(Tensor.full(value, d.clone()));
                    }
                    parts.add(y);
                    if (a > 0) {
                        d[ax] = (int) a;
                        parts.add(Tensor.full(value, d.clone()));
                    }
                    y = Tensor.cat(ax, parts.toArray(new Tensor[0]));
                }
                return one(y);
            }
            int spatial = rank - 2;
            int[] before = new int[spatial];
            int[] after = new int[spatial];
            for (int i = 0; i < spatial; i++) {
                before[i] = (int) full[i + 2];
                after[i] = (int) full[i + 2 + rank];
            }
            F.PadMode m = switch (mode) {
                case "constant" -> F.PadMode.CONSTANT;
                case "reflect" -> F.PadMode.REFLECT;
                case "edge" -> F.PadMode.REPLICATE;
                default -> throw unsupported(n, "pad mode " + mode);
            };
            return one(F.pad(x, before, after, m, value));
        });
        HANDLERS.put("Resize", (n, in) -> {
            Tensor x = t(in, 0);
            if (x.rank() != 4) {
                throw unsupported(n, "resizing a non-4-D tensor");
            }
            int h;
            int w;
            if (present(in, 3) && in.get(3).numel() > 0) {
                long[] sizes = in.get(3).longs();
                h = (int) sizes[sizes.length - 2];
                w = (int) sizes[sizes.length - 1];
            } else {
                float[] scales = in.get(2).hostFloats() != null ? in.get(2).hostFloats() : t(in, 2).toFloatArray();
                h = (int) Math.floor(x.dim(2) * scales[scales.length - 2]);
                w = (int) Math.floor(x.dim(3) * scales[scales.length - 1]);
            }
            String mode = n.attrString("mode", "nearest");
            String ct = n.attrString("coordinate_transformation_mode", "half_pixel");
            if (mode.equals("nearest")) {
                if (!ct.equals("asymmetric") || !n.attrString("nearest_mode", "round_prefer_floor").equals("floor")) {
                    throw unsupported(n, "nearest resize other than asymmetric/floor (PyTorch's)");
                }
                return one(F.interpolate(x, h, w, false, false));
            }
            if (mode.equals("linear")) {
                boolean align = ct.equals("align_corners");
                if (!align && !ct.equals("half_pixel") && !ct.equals("pytorch_half_pixel")) {
                    throw unsupported(n, "linear resize with " + ct);
                }
                return one(F.interpolate(x, h, w, true, align));
            }
            throw unsupported(n, "resize mode " + mode);
        });
        HANDLERS.put("NonZero", (n, in) -> one(Value.of(t(in, 0).nonzero().transpose(0, 1), OnnxModel.INT64)));
    }

    /** A scalar input's value; one known only on the device is read back, ending the current trace. */
    static float scalarOf(Node n, Value v) {
        float[] f = v.hostFloats();
        return f != null ? f[0] : v.tensor(true).toFloatArray()[0];
    }

    static long[] ones(int n) {
        long[] o = new long[n];
        Arrays.fill(o, 1);
        return o;
    }

    static Tensor pool(Node n, Tensor x, boolean max) {
        return pool(n, x, max ? F.PoolMode.MAX : F.PoolMode.AVG, 0.0f);
    }

    /**
     * MaxPool, AveragePool and LpPool. Square, symmetric 2-D pooling without ceil mode takes the
     * 2-D kernels; everything else (sizes per axis, asymmetric padding, ceil mode, dilation, an
     * average that excludes the padding) the general path.
     */
    static Tensor pool(Node n, Tensor x, F.PoolMode mode, float p) {
        int spatial = x.rank() - 2;
        if (spatial < 1 || spatial > 3) {
            throw unsupported(n, spatial + "-D pooling");
        }
        long[] kernel = n.attrInts("kernel_shape", null);
        long[] strides = n.attrInts("strides", ones(spatial));
        long[] dil = n.attrInts("dilations", ones(spatial));
        long[] pads = pads(n, spatial, Arrays.copyOfRange(dimsOf(x), 2, x.rank()), kernel, strides, dil);
        boolean ceil = n.attrInt("ceil_mode", 0) != 0;
        boolean includePad = n.attrInt("count_include_pad", 0) != 0;
        int[] k = toInts(kernel);
        int[] s = toInts(strides);
        int[] pd = toInts(pads);
        int[] d = toInts(dil);
        boolean zeroPad = true;
        for (int v : pd) {
            zeroPad &= v == 0;
        }
        boolean simple = spatial == 2 && !ceil && same(k) && same(s) && same(pd) && same(d) && d[0] == 1 && (mode == F.PoolMode.MAX || includePad || zeroPad);
        if (simple) {
            return mode == F.PoolMode.MAX ? F.maxPool2d(x, k[0], s[0], pd[0]) : F.avgPool2d(x, k[0], s[0], pd[0]);
        }
        return F.pool(x, mode, k, s, pd, d, ceil, includePad, p);
    }

    private static boolean same(int[] v) {
        for (int x : v) {
            if (x != v[0]) {
                return false;
            }
        }
        return true;
    }
}
