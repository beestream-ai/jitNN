package org.beehive.jitnn.bench;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;

/**
 * One distinct operator call of the workloads: the operator, its attributes and its input and
 * output shapes and types. It replays the call on fresh inputs, the same each time, so every call
 * after the first replays one compiled plan: random FP32 in [-1, 1); INT32 inputs with their
 * captured values when those were known on the host (sequence lengths, indices made from
 * constants), else zeros; zero quantised blocks.
 */
final class OpCase {

    final Op op;
    final int[] iattr;
    final float[] fattr;
    final int[][] inShapes;
    final DType[] inTypes;
    final int[][] outShapes;
    final DType[] outTypes;
    final String key;
    /** The values of INT32 inputs that were known on the host when captured (lengths, indices from constants), else null. */
    final int[][] intValues;
    /** How often each workload runs this call per step. */
    final Map<String, Integer> uses = new LinkedHashMap<>();
    private Tensor[] inputs;

    OpCase(Node n) {
        this.op = n.op;
        this.iattr = n.iattr == null ? new int[0] : n.iattr.clone();
        this.fattr = n.fattr == null ? new float[0] : n.fattr.clone();
        inShapes = new int[n.inputs.length][];
        inTypes = new DType[n.inputs.length];
        for (int i = 0; i < n.inputs.length; i++) {
            inShapes[i] = n.inputs[i].shape().dims();
            inTypes[i] = n.inputs[i].dtype();
        }
        intValues = new int[n.inputs.length][];
        for (int i = 0; i < n.inputs.length; i++) {
            Tensor t = n.inputs[i];
            if (t.dtype() == DType.I32 && t.storage().isHostValid()) {
                uk.ac.manchester.tornado.api.types.arrays.IntArray a = (uk.ac.manchester.tornado.api.types.arrays.IntArray) t.storage().host();
                int[] v = new int[t.numel()];
                for (int k = 0; k < v.length; k++) {
                    v[k] = a.get(k);
                }
                intValues[i] = v;
            }
        }
        outShapes = new int[n.outputs.length][];
        outTypes = new DType[n.outputs.length];
        for (int i = 0; i < n.outputs.length; i++) {
            outShapes[i] = n.outputs[i].shape().dims();
            outTypes[i] = n.outputs[i].dtype();
        }
        key = op.name() + "|" + Arrays.toString(iattr) + "|" + Arrays.toString(fattr) + "|" + Arrays.deepToString(inShapes) + Arrays.toString(inTypes) + "|"
                + Arrays.deepToString(outShapes) + Arrays.toString(outTypes);
    }

    static int numel(int[] dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        return n;
    }

    /** Whether every input type can be made up for a replay. */
    boolean replayable() {
        for (DType t : inTypes) {
            if (t == DType.BF16 || t == DType.FP8_E4M3 || t == DType.FP8_E5M2) {
                return false;
            }
        }
        return true;
    }

    private static Tensor input(int[] dims, DType t, long seed) {
        int n = numel(dims);
        return switch (t) {
            case F32 -> Tensor.persistent(Tensor.of(Workloads.uniform(Math.max(n, 1), seed), n == 0 ? new int[] { 1 } : dims));
            case F16 -> Tensor.persistentHalf(Workloads.uniform(n, seed), dims);
            case I32 -> Tensor.of(new int[Math.max(n, 1)], n == 0 ? new int[] { 1 } : dims);
            default -> Tensor.quantized(t, new byte[n / t.blockValues() * t.blockBytes()], dims);
        };
    }

    /** Records the call once (inputs made on the first use); the caller runs it with Engine.sync(). */
    void record() {
        if (inputs == null) {
            inputs = new Tensor[inShapes.length];
            for (int i = 0; i < inputs.length; i++) {
                inputs[i] = intValues[i] != null ? Tensor.of(intValues[i], inShapes[i]) : input(inShapes[i], inTypes[i], 17 + i);
            }
        }
        Tensor[] outs = new Tensor[outShapes.length];
        for (int i = 0; i < outs.length; i++) {
            outs[i] = Tensor.result(org.beehive.jitnn.Shape.of(outShapes[i]), outTypes[i]);
        }
        try (Grad.Scope ignored = Grad.disabled()) {
            Tensor.record(op, inputs, outs, iattr.length == 0 ? null : iattr, fattr.length == 0 ? null : fattr, true);
        }
    }

    void release() {
        inputs = null;
    }

    static long bytesOf(int[] dims, DType t) {
        long n = numel(dims);
        return t.isBlockQuantized() ? n / t.blockValues() * t.blockBytes() : n * t.bytes();
    }

    /** Bytes the call must move at least: every input read once and every output written once. */
    long bytes() {
        long b = 0;
        for (int i = 0; i < inShapes.length; i++) {
            b += bytesOf(inShapes[i], inTypes[i]);
        }
        for (int i = 0; i < outShapes.length; i++) {
            b += bytesOf(outShapes[i], outTypes[i]);
        }
        return b;
    }

    /** Arithmetic of the compute-bound operators (multiply-adds count 2), 0 for the others. */
    double flops() {
        int[] a = iattr;
        return switch (op.name()) {
            case "matmul" -> 2.0 * a[0] * a[1] * a[2] * (a.length > 5 ? a[5] : 1);
            case "linear", "linear_q8_0" -> 2.0 * a[0] * a[1] * a[2];
            case "fp8_gemm", "gguf_matvec" -> 2.0 * a[0] * a[1] * a[2];
            case "conv2d", "conv2d_bwd_data", "conv2d_bwd_weight" -> 2.0 * a[0] * a[4] * a[5] * a[6] * ((double) a[1] / a[11]) * a[7] * a[7];
            case "flash_attention" -> 4.0 * a[0] * a[1] * a[2] * a[3] * (a[4] == 1 ? 0.5 : 1);
            case "flash_attention_bwd" -> 10.0 * a[0] * a[1] * a[2] * a[3] * (a[4] == 1 ? 0.5 : 1);
            default -> 0;
        };
    }

    /** True when the call's matrix operands are half precision (the tensor-core ceiling applies). */
    boolean half() {
        for (DType t : inTypes) {
            if (t == DType.F16 || t == DType.BF16) {
                return true;
            }
        }
        return false;
    }

    static String family(String op) {
        if (op.startsWith("conv") || op.equals("unfold") || op.equals("fold")) {
            return "convolution";
        }
        if (op.startsWith("flash_attention") || op.startsWith("attention_softmax")) {
            return "attention";
        }
        return switch (op) {
            case "matmul", "linear" -> "gemm";
            case "row_norm", "row_norm_bwd", "batch_norm", "batch_norm_bwd", "lrn", "lrn_bwd", "fold_batch_norm" -> "normalisation";
            case "softmax", "softmax_bwd", "cross_entropy", "cross_entropy_bwd", "nll", "nll_bwd" -> "softmax";
            case "reduce", "argmax", "scan", "cumprod_bwd", "grad_stats", "topk", "topk_grad", "sort", "sort_bwd" -> "reduction";
            case "optim", "lamb_moments", "trust_update", "lars_update" -> "optimiser";
            case "embedding", "embedding_bwd", "index_select", "index_select_bwd", "gather", "scatter", "segment_reduce", "segment_reduce_bwd", "one_hot" -> "indexing";
            case "lstm", "lstm_bwd", "gru", "gru_bwd" -> "recurrent";
            case "pool2d", "pool2d_bwd", "adaptive_pool", "adaptive_pool_bwd", "upsample2d", "upsample2d_bwd" -> "pooling";
            case "gguf_matvec", "gguf_dequantize", "linear_q8_0", "quantize_q8_0", "dequantize_q8_0", "fp8_gemm", "fp8_quantize", "fp8_dequantize" -> "quantised";
            default -> "elementwise";
        };
    }
}
