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
package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Emitter;
import ai.beestream.jitnn.engine.Node;
import ai.beestream.jitnn.engine.Op;
import ai.beestream.jitnn.kernels.FlashKernels128;
import ai.beestream.jitnn.kernels.FlashKernels32;
import ai.beestream.jitnn.kernels.FlashKernels64;
import ai.beestream.jitnn.kernels.AttentionGatherKernels;
import ai.beestream.jitnn.kernels.FlashMma;
import ai.beestream.jitnn.kernels.FlashTiled64;
import ai.beestream.jitnn.kernels.FlashTiled64Half;
import ai.beestream.jitnn.kernels.AttentionHalfKernels;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Fused scaled dot-product attention ({@link FlashKernels32}, {@link FlashKernels64} and
 * {@link FlashKernels128} by head dimension): softmax(q k^T * scale + mask) v without storing the
 * scores, in one of three layouts that the kernels read and write through strides:
 * <ul>
 * <li>{@link #BHTD}: q [B, H, T, d], k and v [B, H, S, d], o [B, H, T, d] (also [Z, T, d]);</li>
 * <li>{@link #PACKED}: one projection qkv [B, T, 3 * H * d] holding q, k and v side by side
 * (self-attention), o [B, T, H * d]; the gradient is one packed tensor;</li>
 * <li>{@link #BTHD}: q [B, T, H, d], k and v [B, S, H, d], o [B, T, H, d].</li>
 * </ul>
 * The additive mask broadcasts to [B, H, T, S]. Inputs [q, k, v] or [qkv], then the optional
 * mask; outputs [o, lse]. iattr = [Z, T, S, d, causal, heads, mb, mh, mt, ms, layout]; fattr = [scale].
 */
public final class FlashAttention implements Op {

    public static final int BHTD = 0;
    public static final int PACKED = 1;
    public static final int BTHD = 2;
    /** Largest head dimension of the fused kernels. */
    public static final int MAX_HEAD = FlashKernels128.D;
    /** The tiled kernels ({@link FlashTiled64}) for head dimensions up to 64 unless {@code -Djitnn.flash=simple}. */
    static final boolean TILED = !"simple".equals(System.getProperty("jitnn.flash"));

    static final FlashAttention OP = new FlashAttention();
    static final Backward BACKWARD = new Backward();

    @Override
    public boolean acceptsHalf() {
        return true;
    }

    private FlashAttention() {
    }

    /** True when the fused kernels handle q, k, v in the [B, H, T, d] layout (FP32, v as wide as q). */
    public static boolean supports(Tensor q, Tensor k, Tensor v) {
        return q.dtype() == DType.F32 && k.dtype() == DType.F32 && v.dtype() == DType.F32 && q.rank() >= 3 && q.rank() <= 4 && q.dim(-1) <= MAX_HEAD
                && v.dim(-1) == q.dim(-1) && k.dim(-1) == q.dim(-1) && k.shape().equals(v.shape()) && k.rank() == q.rank();
    }

    /** q [B, H, T, d] (or [Z, T, d]), k and v [.., S, d]: o [B, H, T, d]. */
    public static Tensor apply(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        int t = q.dim(-2);
        int s = k.dim(-2);
        int d = q.dim(-1);
        int heads = q.rank() == 4 ? q.dim(1) : 1;
        return record(new Tensor[] { q, k, v }, q.shape(), q.numel() / (t * d), t, s, d, heads, mask, causal, scale, BHTD);
    }

    /** Self-attention on a packed projection qkv [B, T, 3 * heads * d]: o [B, T, heads * d]. */
    public static Tensor packed(Tensor qkv, int heads, Tensor mask, boolean causal, float scale) {
        Checks.f32(qkv);
        if (qkv.dtype() == DType.F16 && !halfKernel(qkv.dim(2) / 3 / heads)) {
            qkv = qkv.to(DType.F32);
        }
        Checks.rank(qkv, 3);
        int b = qkv.dim(0);
        int t = qkv.dim(1);
        int dim = qkv.dim(2) / 3;
        int d = dim / heads;
        if (3 * dim != qkv.dim(2) || d * heads != dim || d > MAX_HEAD) {
            throw new IllegalArgumentException("packed attention: " + qkv.shape() + " with " + heads + " heads");
        }
        return record(new Tensor[] { qkv }, Shape.of(b, t, dim), b * heads, t, t, d, heads, mask, causal, scale, PACKED);
    }

    /** q [B, T, H, d], k and v [B, S, H, d]: o [B, T, H, d]. */
    public static Tensor bthd(Tensor q, Tensor k, Tensor v, Tensor mask, boolean causal, float scale) {
        Checks.f32(q, k, v);
        Checks.rank(q, 4);
        if (!k.shape().equals(v.shape()) || k.rank() != 4 || k.dim(2) != q.dim(2) || k.dim(3) != q.dim(3) || q.dim(3) > MAX_HEAD) {
            throw new IllegalArgumentException("attention [B, T, H, d]: " + q.shape() + ", " + k.shape() + ", " + v.shape());
        }
        int heads = q.dim(2);
        return record(new Tensor[] { q, k, v }, q.shape(), q.dim(0) * heads, q.dim(1), k.dim(1), q.dim(3), heads, mask, causal, scale, BTHD);
    }

    /** The tensor-core kernels ({@link FlashMma}) for FP16 inputs on CUDA ({@code -Djitnn.flash.mma=false} turns them off). */
    private static final boolean MMA = !"false".equals(System.getProperty("jitnn.flash.mma"));

    /** Whether FP16 attention of head dimension d runs on the tensor-core kernels. */
    private static boolean mmaKernels(int d) {
        return MMA && d <= FlashMma.D && ai.beestream.jitnn.engine.Engine.onCuda() && MatMul.tensorCores();
    }

    /**
     * True when half-precision inference runs FP16 q, k, v through an FP16 kernel (the tiled one for
     * head dimensions up to 64, the tensor-core ones up to 128, on a device); otherwise they are
     * converted to FP32.
     */
    private static boolean halfKernel(int d) {
        return ai.beestream.jitnn.Autocast.halfActivations() && ai.beestream.jitnn.engine.Engine.backend() == ai.beestream.jitnn.engine.Engine.Backend.TORNADO
                && (mmaKernels(d) || d <= FlashKernels64.D && TILED && ai.beestream.jitnn.engine.Engine.localMemoryFits(FlashTiled64.FORWARD_LOCAL_BYTES));
    }

    private static Tensor record(Tensor[] qkv, Shape outShape, int z, int t, int s, int d, int heads, Tensor mask, boolean causal, float scale, int layout) {
        // FP16 stays FP16 for the FP16 kernel, or for MLX's FP16 attention (Metal, where that kernel does not fit)
        boolean mlxHalf = qkv[0].dtype() == DType.F16 && mask == null && ai.beestream.jitnn.Autocast.halfActivations()
                && ai.beestream.jitnn.engine.Engine.backend() == ai.beestream.jitnn.engine.Engine.Backend.TORNADO
                && ai.beestream.jitnn.natives.Libs.mlxHalfAttentionAfterGather(z, t, s, d, causal);
        boolean half = qkv[0].dtype() == DType.F16 && (halfKernel(d) || mlxHalf);
        for (int i = 0; i < qkv.length; i++) {
            if (qkv[i].dtype() == DType.F16 && !half) {
                qkv = qkv.clone();
                qkv[i] = qkv[i].to(DType.F32);
            }
        }
        if (mask != null && mask.dtype() == DType.F16) {
            mask = mask.to(DType.F32);
        }
        int[] strides = { 0, 0, 0, 0 };
        Tensor[] in = qkv;
        if (mask != null) {
            Checks.f32(mask);
            strides = mask.shape().broadcastStrides(Shape.of(z / heads, heads, t, s));
            in = java.util.Arrays.copyOf(qkv, qkv.length + 1);
            in[qkv.length] = mask;
        }
        Tensor o = Tensor.result(outShape, half ? DType.F16 : DType.F32);
        Tensor lse = Tensor.result(Shape.of(z * t), DType.F32);
        int[] iattr = { z, t, s, d, causal ? 1 : 0, heads, strides[0], strides[1], strides[2], strides[3], layout };
        Tensor.record(OP, in, new Tensor[] { o, lse }, iattr, new float[] { scale }, true);
        return o;
    }

    /**
     * FP16 attention in inference ({@code -Djitnn.halfAttention}): {@code auto} takes cuDNN's fused
     * attention, else CUTLASS's batched FP16 GEMMs, when {@code -Djitnn.native} makes them available
     * (a cuDNN attention that builds no plan on the device falls back to CUTLASS), else the FP16
     * FlashAttention kernel; {@code cudnn}, {@code cutlass} and {@code flash} force one.
     */
    private static final String HALF_ATTENTION = System.getProperty("jitnn.halfAttention", "auto");

    private static boolean useCudnnAttention(int t, int s, int d, int causal) {
        return (HALF_ATTENTION.equals("auto") || HALF_ATTENTION.equals("cudnn")) && d % 8 == 0 && d <= 256 && (causal == 0 || t == s)
                && ai.beestream.jitnn.natives.Libs.fusedAttentionAvailable();
    }

    /** q, k, v gathered per problem into FP16 [z, t, d] and [z, s, d] (BHSD), cuDNN's fused attention, O merged into the output's layout. */
    private static void emitCudnnAttention(Emitter e, int[] a, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray q,
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray k, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray v,
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray o, float scale) {
        int z = a[0], t = a[1], s = a[2], d = a[3], causal = a[4], heads = a[5];
        int[] m = meta(a);
        var qz = e.scratchF16(z * t * d);
        var kz = e.scratchF16(z * s * d);
        var vz = e.scratchF16(z * s * d);
        var oz = e.scratchF16(z * t * d);
        int qn = z * t * d;
        // q, k and v in one pass over the projection
        IntArray meta = e.constant(m);
        int total = z * Math.max(t, s) * d;
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, AttentionHalfKernels::gatherQkv, e.context(), q, k, v, qz, kz, vz, meta, t, s, d, total));
        ai.beestream.jitnn.natives.Libs.fusedAttention(e, qz, kz, vz, oz, z / heads, heads, t, s, d, scale, causal != 0);
        e.task(Emitter.grid1D(qn), (g, id) -> g.task(id, AttentionHalfKernels::mergeHalf, e.context(), oz, o, heads, t, d, m[17], m[18], m[19], m[20], qn));
    }

    private static boolean useCutlassAttention(int z, int t, int s, int d) {
        return (HALF_ATTENTION.equals("auto") || HALF_ATTENTION.equals("cutlass")) && s % 4 == 0 && d % 4 == 0 && (long) z * t * s <= (256L << 20)
                && ai.beestream.jitnn.natives.Libs.hgemmBatchedAvailable();
    }

    /**
     * q, v gathered per problem into FP16 [z, t, d] and [z, s, d], k into k^T [z, d, s]; S = q k^T and
     * O = P v as CUTLASS batched FP16 GEMMs; P = softmax(scale S) one warp per row; O merged into the
     * output's layout. No log-sum-exp (inference) and no mask tensor.
     */
    private static void emitCutlassAttention(Emitter e, int[] a, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray q,
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray k, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray v,
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray o, float scale) {
        int z = a[0], t = a[1], s = a[2], d = a[3], causal = a[4], heads = a[5];
        int[] m = meta(a);
        var qz = e.scratchF16(z * t * d);
        var kt = e.scratchF16(z * s * d);
        var vz = e.scratchF16(z * s * d);
        var scores = e.scratchF16(z * t * s);
        var p = e.scratchF16(z * t * s);
        var oz = e.scratchF16(z * t * d);
        int qn = z * t * d;
        int kn = z * s * d;
        int rows = z * t;
        e.task(Emitter.grid1D(qn), (g, id) -> g.task(id, AttentionHalfKernels::gather, e.context(), q, qz, heads, t, d, m[5], m[6], m[7], m[8], qn));
        e.task(Emitter.grid1D(kn), (g, id) -> g.task(id, AttentionHalfKernels::gatherTransposed, e.context(), k, kt, heads, s, d, m[9], m[10], m[11], m[12], kn));
        e.task(Emitter.grid1D(kn), (g, id) -> g.task(id, AttentionHalfKernels::gather, e.context(), v, vz, heads, s, d, m[13], m[14], m[15], m[16], kn));
        ai.beestream.jitnn.natives.Libs.hgemmBatched(e, qz, kt, scores, t, s, d, z);
        e.task(Emitter.blocks1D(Emitter.ceilDiv(rows, 8), 256),
                (g, id) -> g.task(id, AttentionHalfKernels::softmaxHalfWarp, e.context(), scores, p, rows, s, scale, causal, t));
        ai.beestream.jitnn.natives.Libs.hgemmBatched(e, p, vz, oz, t, d, s, z);
        e.task(Emitter.grid1D(qn), (g, id) -> g.task(id, AttentionHalfKernels::mergeHalf, e.context(), oz, o, heads, t, d, m[17], m[18], m[19], m[20], qn));
    }

    /**
     * Packed or [B, T, H, d] q, k, v gathered into [z, t, d] and [z, s, d], a library's fused attention
     * on them (MLX on Metal), O merged back into the output's layout. False when the library declines.
     */
    private static boolean emitGatheredAttention(Emitter e, int[] a, FloatArray q, FloatArray k, FloatArray v, FloatArray o, float scale, FloatArray bias) {
        int z = a[0], t = a[1], s = a[2], d = a[3], causal = a[4], heads = a[5];
        int[] m = meta(a);
        FloatArray qz = e.sharedScratch("attentionQ", z * t * d);
        FloatArray kz = e.sharedScratch("attentionK", z * s * d);
        FloatArray vz = e.sharedScratch("attentionV", z * s * d);
        FloatArray oz = e.sharedScratch("attentionO", z * t * d);
        IntArray meta = e.constant(m);
        int total = z * Math.max(t, s) * d;
        int hasBias = bias != null ? 1 : 0;
        FloatArray b = bias != null ? bias : e.dummyF32();
        int rowLen = 3 * heads * d;
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, AttentionGatherKernels::gatherQkv, e.context(), q, k, v, qz, kz, vz, meta, t, s, d, total, b, rowLen, hasBias));
        if (!ai.beestream.jitnn.natives.Libs.attention(e, qz, kz, vz, oz, z / heads, heads, t, s, d, scale, causal != 0)) {
            return false;
        }
        int qn = z * t * d;
        e.task(Emitter.grid1D(qn), (g, id) -> g.task(id, AttentionGatherKernels::merge, e.context(), oz, o, heads, t, d, m[17], m[18], m[19], m[20], qn));
        return true;
    }

    /** {@link #emitGatheredAttention} on FP16 tensors with MLX's FP16 attention (an FP16 bias). */
    private static boolean emitGatheredAttentionHalf(Emitter e, int[] a, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, HalfFloatArray o, float scale,
            HalfFloatArray bias) {
        int z = a[0], t = a[1], s = a[2], d = a[3], causal = a[4], heads = a[5];
        int[] m = meta(a);
        HalfFloatArray qz = e.sharedScratchF16("attentionQ", z * t * d);
        HalfFloatArray kz = e.sharedScratchF16("attentionK", z * s * d);
        HalfFloatArray vz = e.sharedScratchF16("attentionV", z * s * d);
        HalfFloatArray oz = e.sharedScratchF16("attentionO", z * t * d);
        IntArray meta = e.constant(m);
        int total = z * Math.max(t, s) * d;
        int hasBias = bias != null ? 1 : 0;
        HalfFloatArray b = bias != null ? bias : e.scratchF16(1);
        int rowLen = 3 * heads * d;
        e.task(Emitter.grid1D(total), (g, id) -> g.task(id, AttentionGatherKernels::gatherQkvHalf, e.context(), q, k, v, qz, kz, vz, meta, t, s, d, total, b, rowLen, hasBias));
        if (!ai.beestream.jitnn.natives.Libs.mlxAttentionHalf(e, qz, kz, vz, oz, z / heads, heads, t, s, d, scale, causal != 0)) {
            return false;
        }
        int qn = z * t * d;
        e.task(Emitter.grid1D(qn), (g, id) -> g.task(id, AttentionHalfKernels::mergeHalf, e.context(), oz, o, heads, t, d, m[17], m[18], m[19], m[20], qn));
        return true;
    }

    /** iattr index set when the last input is the bias of the projection that produced the packed q, k, v (added as they are read). */
    static final int QKV_BIAS = 11;

    /**
     * A projection's bias moved into the attention that reads its packed q, k, v (the projection's only
     * reader), where the gather for a library's fused attention adds it: the bias pass goes away.
     * Only where that gather runs (MLX on Metal), with no mask and the log-sum-exp unread.
     */
    public static java.util.List<Node> foldQkvBias(java.util.List<Node> nodes, java.util.Set<ai.beestream.jitnn.engine.Storage> wanted) {
        java.util.IdentityHashMap<ai.beestream.jitnn.engine.Storage, Integer> uses = new java.util.IdentityHashMap<>();
        java.util.IdentityHashMap<ai.beestream.jitnn.engine.Storage, Integer> reader = new java.util.IdentityHashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            for (Tensor x : nodes.get(i).inputs) {
                uses.merge(x.storage(), 1, Integer::sum);
                reader.put(x.storage(), i);
            }
        }
        java.util.List<Node> out = null;
        for (int i = 0; i < nodes.size(); i++) {
            Node l = nodes.get(i);
            if (l.op != Linear.OP || l.iattr[3] == 0 || l.iattr[4] != Linear.NONE || wanted.contains(l.out(0).storage())) {
                continue;
            }
            Integer j = reader.get(l.out(0).storage());
            if (j == null || j <= i || uses.getOrDefault(l.out(0).storage(), 0) != 1) {
                continue;
            }
            Node f = nodes.get(j);
            if (f.op != OP || f.iattr[10] != PACKED || f.inputs.length != 1 || f.in(0).storage() != l.out(0).storage() || uses.containsKey(f.out(1).storage())
                    || wanted.contains(f.out(1).storage()) || f.in(0).dtype() != l.in(2).dtype()) {
                continue;
            }
            int[] fa = f.iattr;
            boolean half = f.out(0).dtype() == DType.F16;
            boolean mlx = half ? ai.beestream.jitnn.natives.Libs.mlxHalfAttentionAfterGather(fa[0], fa[1], fa[2], fa[3], fa[4] != 0)
                    : ai.beestream.jitnn.natives.Libs.mlxAttentionAfterGather(fa[0], fa[1], fa[2], fa[3], fa[4] != 0);
            if (!mlx) {
                continue;
            }
            if (out == null) {
                out = new java.util.ArrayList<>(nodes);
            }
            int[] la = l.iattr.clone();
            la[3] = 0;
            out.set(i, new Node(Linear.OP, new Tensor[] { l.in(0), l.in(1) }, l.outputs, la, l.fattr, true));
            int[] na = java.util.Arrays.copyOf(fa, QKV_BIAS + 1);
            na[QKV_BIAS] = 1;
            out.set(j, new Node(OP, new Tensor[] { f.in(0), l.in(2) }, f.outputs, na, f.fattr, true));
        }
        return out == null ? nodes : out;
    }

    /** [heads, the mask strides, then the (offset, batch, head, token) strides of q, k, v and o]. */
    static int[] meta(int[] a) {
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int heads = a[5];
        int dim = heads * d;
        int[] q;
        int[] k;
        int[] v;
        int[] o;
        switch (a[10]) {
            case PACKED -> {
                q = new int[] { 0, t * 3 * dim, d, 3 * dim };
                k = new int[] { dim, t * 3 * dim, d, 3 * dim };
                v = new int[] { 2 * dim, t * 3 * dim, d, 3 * dim };
                o = new int[] { 0, t * dim, d, dim };
            }
            case BTHD -> {
                q = new int[] { 0, t * dim, d, dim };
                k = new int[] { 0, s * dim, d, dim };
                v = k.clone();
                o = q.clone();
            }
            default -> {
                q = new int[] { 0, heads * t * d, t * d, d };
                k = new int[] { 0, heads * s * d, s * d, d };
                v = k.clone();
                o = q.clone();
            }
        }
        int[] m = new int[21];
        m[0] = heads;
        System.arraycopy(a, 6, m, 1, 4);
        System.arraycopy(q, 0, m, 5, 4);
        System.arraycopy(k, 0, m, 9, 4);
        System.arraycopy(v, 0, m, 13, 4);
        System.arraycopy(o, 0, m, 17, 4);
        return m;
    }

    /** Index of row {@code row} of problem z in operand {@code which} (0 q, 1 k, 2 v, 3 o). */
    static int at(int[] meta, int which, int z, int row) {
        int b = z / meta[0];
        int h = z - b * meta[0];
        int m = 5 + 4 * which;
        return meta[m] + b * meta[m + 1] + h * meta[m + 2] + row * meta[m + 3];
    }

    @Override
    public String name() {
        return "flash_attention";
    }

    private static int rowsPerGroup(int d) {
        return d <= FlashKernels32.D ? FlashKernels32.ROWS : (d <= FlashKernels64.D ? FlashKernels64.ROWS : FlashKernels128.ROWS);
    }

    private static WorkerGrid grid(int z, int rows, int d) {
        return Emitter.blocks1D(z * Emitter.ceilDiv(rows, rowsPerGroup(d)), FlashKernels32.THREADS);
    }

    @Override
    public void emit(Emitter e, Node n) {
        int[] a = n.iattr;
        int z = a[0];
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int causal = a[4];
        float scale = n.fattr[0];
        IntArray meta = e.constant(meta(a));
        boolean packed = a[10] == PACKED;
        int qkvCount = packed ? 1 : 3;
        boolean qkvBias = a.length > QKV_BIAS && a[QKV_BIAS] != 0;
        int hm = n.inputs.length - qkvCount - (qkvBias ? 1 : 0) > 0 ? 1 : 0;
        Tensor biasT = qkvBias ? n.in(n.inputs.length - 1) : null;
        int rows = z / a[5] * t;
        int cols = 3 * a[5] * d;
        if (n.out(0).dtype() == DType.F16) {
            var qh = e.f16(n.in(0));
            var kh = packed ? qh : e.f16(n.in(1));
            var vh = packed ? qh : e.f16(n.in(2));
            FloatArray mk = hm != 0 ? e.f32(n.in(qkvCount)) : e.dummyF32();
            var oh = e.f16(n.out(0));
            if (hm == 0 && e.unread(n.out(1)) && ai.beestream.jitnn.natives.Libs.mlxHalfAttentionAfterGather(z, t, s, d, causal != 0)
                    && emitGatheredAttentionHalf(e, a, qh, kh, vh, oh, scale, biasT != null ? e.f16(biasT) : null)) {
                return;
            }
            if (biasT != null) {
                // the paths below read q, k, v as they are: the projection's bias goes on first
                FloatArray bf = biasT.dtype() == DType.F16 ? halfToFloat(e, e.f16(biasT), cols) : e.f32(biasT);
                int total = rows * cols;
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ai.beestream.jitnn.kernels.LibraryGlueKernels::epilogueHalf, e.context(), qh, bf, rows, cols, total,
                        ai.beestream.jitnn.kernels.GemmKernels.BIAS));
            }
            if (hm == 0 && e.unread(n.out(1))) {
                if (useCudnnAttention(t, s, d, causal)) {
                    if (packed && t == s && ai.beestream.jitnn.natives.Libs.packedAttentionAvailable(a[5], d)) {
                        // cuDNN reads q, k and v in place from the projection and writes the merged heads
                        ai.beestream.jitnn.natives.Libs.attentionPacked(e, qh, oh, z / a[5], a[5], t, d, scale, causal != 0);
                        return;
                    }
                    emitCudnnAttention(e, a, qh, kh, vh, oh, scale);
                    return;
                }
                if (useCutlassAttention(z, t, s, d)) {
                    emitCutlassAttention(e, a, qh, kh, vh, oh, scale);
                    return;
                }
            }
            if (!ai.beestream.jitnn.engine.Engine.localMemoryFits(FlashTiled64.FORWARD_LOCAL_BYTES)) {
                // the FP16 kernel does not fit (Metal): the FP32 kernels on converted copies
                FloatArray qf = halfToFloat(e, qh, n.in(0).numel());
                FloatArray kf = packed ? qf : halfToFloat(e, kh, n.in(1).numel());
                FloatArray vf = packed ? qf : halfToFloat(e, vh, n.in(2).numel());
                int on = n.out(0).numel();
                FloatArray of = e.scratch(on);
                emitKernels(e, n, a, qf, kf, vf, mk, of, hm, meta, scale);
                e.task(Emitter.grid1D(on), (g, id) -> g.task(id, ai.beestream.jitnn.kernels.LowPrecision::f32ToF16, e.context(), of, oh, on));
                return;
            }
            FloatArray lh = e.f32(n.out(1));
            if (mmaKernels(d)) {
                int size = FlashMma.size(d);
                int qr = FlashMma.forwardRows(size);
                WorkerGrid mg = Emitter.blocks1D(z * Emitter.ceilDiv(t, qr), 2 * qr);
                switch (size) {
                    case 32 -> e.task(mg, (g, id) -> g.task(id, FlashMma::forward32, e.context(), qh, kh, vh, mk, oh, lh, t, s, d, scale, causal, hm, meta));
                    case 64 -> e.task(mg, (g, id) -> g.task(id, FlashMma::forward64, e.context(), qh, kh, vh, mk, oh, lh, t, s, d, scale, causal, hm, meta));
                    default -> e.task(mg, (g, id) -> g.task(id, FlashMma::forward128, e.context(), qh, kh, vh, mk, oh, lh, t, s, d, scale, causal, hm, meta));
                }
                return;
            }
            WorkerGrid tiled = Emitter.blocks2D(z * Emitter.ceilDiv(t, FlashTiled64.BQ), 1, 16, 16);
            e.task(tiled, (g, id) -> g.task(id, FlashTiled64Half::forward, e.context(), qh, kh, vh, mk, oh, lh, t, s, d, scale, causal, hm, meta));
            return;
        }
        FloatArray q = e.f32(n.in(0));
        FloatArray k = packed ? q : e.f32(n.in(1));
        FloatArray v = packed ? q : e.f32(n.in(2));
        FloatArray mask = hm != 0 ? e.f32(n.in(qkvCount)) : e.dummyF32();
        FloatArray o = e.f32(n.out(0));
        if (hm == 0 && a[10] == BHTD && e.unread(n.out(1))
                && ai.beestream.jitnn.natives.Libs.attention(e, q, k, v, o, z / a[5], a[5], t, s, d, scale, causal != 0)) {
            return;
        }
        FloatArray bias = biasT != null ? e.f32(biasT) : null;
        if (hm == 0 && a[10] != BHTD && e.unread(n.out(1)) && ai.beestream.jitnn.natives.Libs.mlxAttentionAfterGather(z, t, s, d, causal != 0)
                && emitGatheredAttention(e, a, q, k, v, o, scale, bias)) {
            return;
        }
        if (bias != null) {
            int total = rows * cols;
            e.task(Emitter.grid1D(total), (g, id) -> g.task(id, ai.beestream.jitnn.kernels.LibraryGlueKernels::epilogueBatched, e.context(), q, bias, rows, cols, total,
                    ai.beestream.jitnn.kernels.GemmKernels.BIAS));
        }
        emitKernels(e, n, a, q, k, v, mask, o, hm, meta, scale);
    }

    /** An FP32 copy of the first count elements of an FP16 array. */
    private static FloatArray halfToFloat(Emitter e, HalfFloatArray h, int count) {
        FloatArray f = e.scratch(count);
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, ai.beestream.jitnn.kernels.LowPrecision::f16ToF32, e.context(), h, f, count));
        return f;
    }

    /** jitNN's FP32 attention kernels, by head dimension. */
    private static void emitKernels(Emitter e, Node n, int[] a, FloatArray q, FloatArray k, FloatArray v, FloatArray mask, FloatArray o, int hasMask, IntArray meta, float scale) {
        int z = a[0];
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int causal = a[4];
        FloatArray lse = e.f32(n.out(1));
        WorkerGrid g0 = grid(z, t, d);
        if (d <= FlashKernels32.D && !(TILED && ai.beestream.jitnn.engine.Engine.localMemoryFits(FlashTiled64.FORWARD_LOCAL_BYTES))) {
            e.task(g0, (g, id) -> g.task(id, FlashKernels32::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else if (d <= FlashKernels64.D && TILED && ai.beestream.jitnn.engine.Engine.localMemoryFits(FlashTiled64.FORWARD_LOCAL_BYTES)) {
            // the tiled kernel needs 43.5 KB of local memory: more than Metal's 32 KB
            WorkerGrid tiled = Emitter.blocks2D(z * Emitter.ceilDiv(t, FlashTiled64.BQ), 1, 16, 16);
            e.task(tiled, (g, id) -> g.task(id, FlashTiled64::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else if (d <= FlashKernels64.D) {
            e.task(g0, (g, id) -> g.task(id, FlashKernels64::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        } else {
            e.task(g0, (g, id) -> g.task(id, FlashKernels128::forward, e.context(), q, k, v, mask, o, lse, t, s, d, scale, causal, hasMask, meta));
        }
    }

    /** Host views of q, k, v and the mask, read through the layout. */
    private record Host(int[] meta, FloatArray q, FloatArray k, FloatArray v, FloatArray mask) {

        static Host of(Node n, int maskAt) {
            boolean packed = n.iattr[10] == PACKED;
            FloatArray q = (FloatArray) n.in(0).storage().host();
            FloatArray k = packed ? q : (FloatArray) n.in(1).storage().host();
            FloatArray v = packed ? q : (FloatArray) n.in(2).storage().host();
            FloatArray mask = maskAt >= 0 ? (FloatArray) n.in(maskAt).storage().host() : null;
            return new Host(FlashAttention.meta(n.iattr), q, k, v, mask);
        }

        double get(int which, int z, int row, int c) {
            FloatArray arr = which == 0 ? q : (which == 1 ? k : v);
            return arr.get(at(meta, which, z, row) + c);
        }

        /** Masked, scaled scores of query row i of problem z in double; NEGATIVE_INFINITY where masked. */
        double[] scores(int[] a, double scale, int z, int i) {
            int s = a[2];
            int d = a[3];
            boolean causal = a[4] != 0;
            double[] row = new double[s];
            for (int j = 0; j < s; j++) {
                if (causal && j > i) {
                    row[j] = Double.NEGATIVE_INFINITY;
                    continue;
                }
                double dot = 0;
                for (int c = 0; c < d; c++) {
                    dot += get(0, z, i, c) * get(1, z, j, c);
                }
                row[j] = dot * scale;
                if (mask != null) {
                    int b = z / a[5];
                    int h = z - b * a[5];
                    row[j] += mask.get(b * a[6] + h * a[7] + i * a[8] + j * a[9]);
                }
            }
            return row;
        }
    }

    /** Softmax of a score row in double. */
    static double[] softmax(double[] row) {
        double m = Double.NEGATIVE_INFINITY;
        for (double x : row) {
            m = Math.max(m, x);
        }
        double sum = 0;
        double[] p = new double[row.length];
        for (int j = 0; j < row.length; j++) {
            p[j] = row[j] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(row[j] - m);
            sum += p[j];
        }
        for (int j = 0; j < row.length; j++) {
            p[j] /= sum;
        }
        return p;
    }

    @Override
    public boolean cpu(Node n) {
        int[] a = n.iattr;
        if (a.length > QKV_BIAS && a[QKV_BIAS] != 0) {
            return false;
        }
        boolean packed = a[10] == PACKED;
        int qkv = packed ? 1 : 3;
        Tensor q = n.in(0);
        return CpuKernels.attention(a, n.fattr[0], q, packed ? q : n.in(1), packed ? q : n.in(2), n.inputs.length > qkv ? n.in(qkv) : null, n.out(0), n.out(1));
    }

    @Override
    public void reference(Node n) {
        int[] a = n.iattr;
        int z = a[0];
        int t = a[1];
        int s = a[2];
        int d = a[3];
        int qkvCount = a[10] == PACKED ? 1 : 3;
        Host h = Host.of(n, n.inputs.length > qkvCount ? qkvCount : -1);
        FloatArray o = (FloatArray) n.out(0).storage().host();
        FloatArray lse = (FloatArray) n.out(1).storage().host();
        for (int zi = 0; zi < z; zi++) {
            for (int i = 0; i < t; i++) {
                double[] row = h.scores(a, n.fattr[0], zi, i);
                double m = Double.NEGATIVE_INFINITY;
                for (double x : row) {
                    m = Math.max(m, x);
                }
                double sum = 0;
                for (double x : row) {
                    sum += x == Double.NEGATIVE_INFINITY ? 0 : Math.exp(x - m);
                }
                double[] p = softmax(row);
                for (int c = 0; c < d; c++) {
                    double acc = 0;
                    for (int j = 0; j < s; j++) {
                        acc += p[j] * h.get(2, zi, j, c);
                    }
                    o.set(at(h.meta(), 3, zi, i) + c, (float) acc);
                }
                lse.set(zi * t + i, (float) (m + Math.log(sum)));
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        int qkvCount = n.iattr[10] == PACKED ? 1 : 3;
        boolean hasMask = n.inputs.length > qkvCount;
        Tensor[] grads = new Tensor[qkvCount];
        for (int i = 0; i < qkvCount; i++) {
            grads[i] = Tensor.result(n.in(i).shape(), DType.F32);
        }
        Tensor[] in = java.util.Arrays.copyOf(n.inputs, n.inputs.length + 3);
        in[n.inputs.length] = n.out(0);
        in[n.inputs.length + 1] = n.out(1);
        in[n.inputs.length + 2] = gy.dtype() == DType.F32 ? gy : gy.to(DType.F32);
        // FP16 activations: the tiled and tensor-core backward kernels read FP16 q, k, v and o as they are; otherwise FP32 copies
        boolean halfIn = in[0].dtype() == DType.F16 && (tiledBackward(n.iattr[3]) || mmaKernels(n.iattr[3]))
                && ai.beestream.jitnn.engine.Engine.backend() == ai.beestream.jitnn.engine.Engine.Backend.TORNADO;
        for (int i = 0; i < in.length; i++) {
            boolean qkvOrO = i < qkvCount || i == n.inputs.length;
            if (in[i].dtype() == DType.F16 && !(halfIn && qkvOrO)) {
                in[i] = in[i].to(DType.F32);
            }
        }
        int[] iattr = java.util.Arrays.copyOf(n.iattr, 12);
        iattr[11] = hasMask ? 1 : 0;
        Tensor.record(BACKWARD, in, grads, iattr, n.fattr.clone(), true);
        return java.util.Arrays.copyOf(grads, n.inputs.length);
    }

    /** Whether the backward of head dimension d runs the tiled kernels (FlashTiled64), which also come in an FP16-input form. */
    static boolean tiledBackward(int d) {
        return d <= FlashKernels64.D && TILED && ai.beestream.jitnn.engine.Engine.localMemoryFits(FlashTiled64.BACKWARD_LOCAL_BYTES);
    }

    /**
     * Inputs [q, k, v | qkv, (mask), o, lse, dO]; outputs [dq, dk, dv] or [dqkv]. iattr as the forward plus [hasMask].
     * q, k, v and o may be FP16 (FP16 activations) for the tiled kernels; the gradients are FP32.
     */
    static final class Backward implements Op {

        @Override
        public boolean acceptsHalf() {
            return true;
        }

        @Override
        public String name() {
            return "flash_attention_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            int z = a[0];
            int t = a[1];
            int s = a[2];
            int d = a[3];
            int causal = a[4];
            int hasMask = a[11];
            float scale = n.fattr[0];
            IntArray meta = e.constant(meta(a));
            boolean packed = a[10] == PACKED;
            int qkvCount = packed ? 1 : 3;
            int at = qkvCount + hasMask;
            FloatArray mask = hasMask != 0 ? e.f32(n.in(qkvCount)) : e.dummyF32();
            FloatArray lse = e.f32(n.in(at + 1));
            FloatArray dout = e.f32(n.in(at + 2));
            FloatArray dq = e.f32(n.out(0));
            FloatArray dk = packed ? dq : e.f32(n.out(1));
            FloatArray dv = packed ? dq : e.f32(n.out(2));
            int rows = z * t;
            FloatArray delta = e.scratch(rows);
            if (n.in(0).dtype() == DType.F16) {
                var qh = e.f16(n.in(0));
                var kh = packed ? qh : e.f16(n.in(1));
                var vh = packed ? qh : e.f16(n.in(2));
                var oh = e.f16(n.in(at));
                if (mmaKernels(d)) {
                    int size = FlashMma.size(d);
                    int qr = FlashMma.backwardQRows(size);
                    int kr = FlashMma.backwardKVRows(size);
                    WorkerGrid mq = Emitter.blocks1D(z * Emitter.ceilDiv(t, qr), 2 * qr);
                    WorkerGrid mk = Emitter.blocks1D(z * Emitter.ceilDiv(s, kr), 2 * kr);
                    switch (size) {
                        case 32 -> {
                            e.task(mq, (g, id) -> g.task(id, FlashMma::backwardQ32, e.context(), qh, kh, vh, mask, oh, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                            e.task(mk, (g, id) -> g.task(id, FlashMma::backwardKV32, e.context(), qh, kh, vh, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
                        }
                        case 64 -> {
                            e.task(mq, (g, id) -> g.task(id, FlashMma::backwardQ64, e.context(), qh, kh, vh, mask, oh, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                            e.task(mk, (g, id) -> g.task(id, FlashMma::backwardKV64, e.context(), qh, kh, vh, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
                        }
                        default -> {
                            e.task(mq, (g, id) -> g.task(id, FlashMma::backwardQ128, e.context(), qh, kh, vh, mask, oh, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                            e.task(mk, (g, id) -> g.task(id, FlashMma::backwardKV128, e.context(), qh, kh, vh, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
                        }
                    }
                    return;
                }
                WorkerGrid tq = Emitter.blocks2D(z * Emitter.ceilDiv(t, 32), 1, 16, 16);
                WorkerGrid tk = Emitter.blocks2D(z * Emitter.ceilDiv(s, 32), 1, 16, 16);
                e.task(tq, (g, id) -> g.task(id, FlashTiled64Half::backwardQ, e.context(), qh, kh, vh, mask, oh, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(tk, (g, id) -> g.task(id, FlashTiled64Half::backwardKV, e.context(), qh, kh, vh, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
                return;
            }
            FloatArray q = e.f32(n.in(0));
            FloatArray k = packed ? q : e.f32(n.in(1));
            FloatArray v = packed ? q : e.f32(n.in(2));
            FloatArray o = e.f32(n.in(at));
            if (d <= FlashKernels64.D && TILED && ai.beestream.jitnn.engine.Engine.localMemoryFits(FlashTiled64.BACKWARD_LOCAL_BYTES)) {
                // backwardQ computes delta for its queries, then backwardKV reads it
                WorkerGrid tq = Emitter.blocks2D(z * Emitter.ceilDiv(t, 32), 1, 16, 16);
                WorkerGrid tk = Emitter.blocks2D(z * Emitter.ceilDiv(s, 32), 1, 16, 16);
                e.task(tq, (g, id) -> g.task(id, FlashTiled64::backwardQ, e.context(), q, k, v, mask, o, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(tk, (g, id) -> g.task(id, FlashTiled64::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
                return;
            }
            e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, FlashKernels64::delta, e.context(), dout, o, delta, meta, t, d, rows));
            WorkerGrid qGrid = grid(z, t, d);
            WorkerGrid kGrid = grid(z, s, d);
            if (d <= FlashKernels32.D) {
                e.task(qGrid, (g, id) -> g.task(id, FlashKernels32::backwardQ, e.context(), q, k, v, mask, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(kGrid, (g, id) -> g.task(id, FlashKernels32::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
            } else if (d <= FlashKernels64.D) {
                e.task(qGrid, (g, id) -> g.task(id, FlashKernels64::backwardQ, e.context(), q, k, v, mask, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(kGrid, (g, id) -> g.task(id, FlashKernels64::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
            } else {
                e.task(qGrid, (g, id) -> g.task(id, FlashKernels128::backwardQ, e.context(), q, k, v, mask, dout, lse, delta, dq, t, s, d, scale, causal, hasMask, meta));
                e.task(kGrid, (g, id) -> g.task(id, FlashKernels128::backwardKV, e.context(), q, k, v, mask, dout, lse, delta, dk, dv, t, s, d, scale, causal, hasMask, meta));
            }
        }

        @Override
        public boolean cpu(Node n) {
            int[] a = n.iattr;
            boolean packed = a[10] == PACKED;
            int qkv = packed ? 1 : 3;
            int at = qkv + a[11];
            Tensor q = n.in(0);
            Tensor dq = n.out(0);
            return CpuKernels.attentionBackward(a, n.fattr[0], q, packed ? q : n.in(1), packed ? q : n.in(2), a[11] != 0 ? n.in(qkv) : null, n.in(at), n.in(at + 1), n.in(at + 2), dq,
                    packed ? dq : n.out(1), packed ? dq : n.out(2));
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            int z = a[0];
            int t = a[1];
            int s = a[2];
            int d = a[3];
            double scale = n.fattr[0];
            boolean packed = a[10] == PACKED;
            int qkvCount = packed ? 1 : 3;
            int at = qkvCount + a[11];
            Host h = Host.of(n, a[11] != 0 ? qkvCount : -1);
            int[] meta = h.meta();
            FloatArray dout = (FloatArray) n.in(at + 2).storage().host();
            FloatArray[] out = new FloatArray[3];
            for (int w = 0; w < 3; w++) {
                out[w] = (FloatArray) n.out(packed ? 0 : w).storage().host();
            }
            double[][] acc = { new double[out[0].getSize()], new double[out[1].getSize()], new double[out[2].getSize()] };
            for (int zi = 0; zi < z; zi++) {
                for (int i = 0; i < t; i++) {
                    double[] p = softmax(h.scores(a, scale, zi, i));
                    double[] dp = new double[s];
                    double dot = 0;
                    for (int j = 0; j < s; j++) {
                        for (int c = 0; c < d; c++) {
                            double g = dout.get(at(meta, 3, zi, i) + c);
                            dp[j] += g * h.get(2, zi, j, c);
                            acc[2][at(meta, 2, zi, j) + c] += p[j] * g;
                        }
                        dot += p[j] * dp[j];
                    }
                    for (int j = 0; j < s; j++) {
                        double g = p[j] * (dp[j] - dot) * scale;
                        for (int c = 0; c < d; c++) {
                            acc[0][at(meta, 0, zi, i) + c] += g * h.get(1, zi, j, c);
                            acc[1][at(meta, 1, zi, j) + c] += g * h.get(0, zi, i, c);
                        }
                    }
                }
            }
            if (packed) {
                // q, k and v occupy disjoint columns of the one gradient tensor
                for (int i = 0; i < out[0].getSize(); i++) {
                    out[0].set(i, (float) (acc[0][i] + acc[1][i] + acc[2][i]));
                }
            } else {
                for (int w = 0; w < 3; w++) {
                    for (int i = 0; i < out[w].getSize(); i++) {
                        out[w].set(i, (float) acc[w][i]);
                    }
                }
            }
        }
    }
}
