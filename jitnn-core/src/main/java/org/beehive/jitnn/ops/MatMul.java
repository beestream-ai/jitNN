package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.GemmKernels;
import org.beehive.jitnn.kernels.MmaGemmKernels;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** 2-D matrix product with optional transposes of either operand. iattr = [M, N, K, tA, tB]. */
public final class MatMul implements Op {

    static final MatMul OP = new MatMul();

    private MatMul() {
    }

    /**
     * op(a) op(b) for matrices, or batched (FP32 result; half-precision operands of the same type
     * are read as such and accumulated in FP32): a [..., M, K] and b [..., K, N] (op applied to the last
     * two axes) with equal batch dimensions, or with one side a plain matrix shared by every batch.
     */
    public static Tensor apply(Tensor a, Tensor b, boolean transA, boolean transB) {
        HostValues.gemmOperands("matmul", a, b);
        if (a.rank() < 2 || b.rank() < 2) {
            throw new IllegalArgumentException("matmul needs matrices, got " + a.shape() + " and " + b.shape());
        }
        int m = transA ? a.dim(-1) : a.dim(-2);
        int ka = transA ? a.dim(-2) : a.dim(-1);
        int kb = transB ? b.dim(-1) : b.dim(-2);
        int n = transB ? b.dim(-2) : b.dim(-1);
        if (ka != kb) {
            throw new IllegalArgumentException("matmul inner dimensions differ: " + a.shape() + (transA ? "^T" : "") + " x " + b.shape() + (transB ? "^T" : ""));
        }
        int[] batchDims;
        if (a.rank() == 2 && b.rank() == 2) {
            batchDims = new int[0];
        } else if (b.rank() == 2) {
            batchDims = java.util.Arrays.copyOf(a.shape().dims(), a.rank() - 2);
        } else if (a.rank() == 2) {
            batchDims = java.util.Arrays.copyOf(b.shape().dims(), b.rank() - 2);
        } else {
            batchDims = java.util.Arrays.copyOf(a.shape().dims(), a.rank() - 2);
            if (!java.util.Arrays.equals(batchDims, java.util.Arrays.copyOf(b.shape().dims(), b.rank() - 2))) {
                throw new IllegalArgumentException("batched matmul needs equal batch dimensions: " + a.shape() + " x " + b.shape());
            }
        }
        int batch = 1;
        for (int d : batchDims) {
            batch *= d;
        }
        int[] dims = java.util.Arrays.copyOf(batchDims, batchDims.length + 2);
        dims[batchDims.length] = m;
        dims[batchDims.length + 1] = n;
        Tensor out = Tensor.result(Shape.of(dims), DType.F32);
        int strideA = a.rank() == 2 ? 0 : m * ka;
        int strideB = b.rank() == 2 ? 0 : ka * n;
        int[] iattr = batchDims.length == 0 ? new int[] { m, n, ka, transA ? 1 : 0, transB ? 1 : 0 }
                : new int[] { m, n, ka, transA ? 1 : 0, transB ? 1 : 0, batch, strideA, strideB };
        Tensor.record(OP, new Tensor[] { a, b }, new Tensor[] { out }, iattr, null, true);
        return out;
    }

    @Override
    public String name() {
        return "matmul";
    }

    /**
     * Number of K slices for C = A^T B when the output alone fills too few work-groups (weight
     * gradients: a long reduction over tokens into a small matrix), or 1. {@code -Djitnn.splitK=false} disables it.
     */
    static int splitK(int m, int n, int k) {
        if (!SPLIT_K) {
            return 1;
        }
        int tiles = Emitter.ceilDiv(m, GemmKernels.BM) * Emitter.ceilDiv(n, GemmKernels.BN);
        int slices = 1;
        while (tiles * slices < 128 && slices < 32 && k % (slices * 2) == 0 && k / (slices * 2) >= 64) {
            slices *= 2;
        }
        return slices;
    }

    private static final boolean SPLIT_K = !"false".equals(System.getProperty("jitnn.splitK"));

    /**
     * Split-K for C = A^T B (A [K, M], B [K, N]): K is the leading dimension of both, so slice z
     * is a contiguous block, and the batched GEMM computes one partial product per slice into a
     * scratch buffer; the partials are then summed in order (deterministic).
     */
    private static void emitSplitK(Emitter e, Node node, int slices) {
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        int ks = k / slices;
        FloatArray c = e.f32(node.out(0));
        FloatArray part = e.scratch(slices * m * n);
        FloatArray bias = e.dummyF32();
        int tilesY = Emitter.ceilDiv(m, GemmKernels.BM);
        WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, GemmKernels.BN), tilesY * slices, 16, 16);
        int strideA = ks * m;
        int strideB = ks * n;
        DType dt = node.in(0).dtype();
        if (dt != DType.F32 && useTensorCores(e)) {
            emitMma(e, node.in(0), node.in(1), part, bias, m, n, ks, 1, 0, 0, slices, strideA, strideB);
        } else if (dt == DType.F16) {
            var a = e.f16(node.in(0));
            var b = e.f16(node.in(1));
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64BatchedF16, e.context(), a, b, part, bias, m, n, ks, 1, 0, 0, strideA, strideB, tilesY));
        } else if (dt == DType.BF16) {
            var a = e.bf16(node.in(0));
            var b = e.bf16(node.in(1));
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64BatchedBF16, e.context(), a, b, part, bias, m, n, ks, 1, 0, 0, strideA, strideB, tilesY));
        } else {
            FloatArray a = e.f32(node.in(0));
            FloatArray b = e.f32(node.in(1));
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64Batched, e.context(), a, b, part, bias, m, n, ks, 1, 0, 0, strideA, strideB, tilesY));
        }
        int cells = m * n;
        e.task(Emitter.grid1D(cells), (g, id) -> g.task(id, org.beehive.jitnn.kernels.NormKernels::sumPartials, e.context(), part, c, slices, cells));
    }

    @Override
    public void emit(Emitter e, Node node) {
        if (node.iattr.length == 5 && node.iattr[3] == 1 && node.iattr[4] == 0) {
            int slices = splitK(node.iattr[0], node.iattr[1], node.iattr[2]);
            if (slices > 1) {
                emitSplitK(e, node, slices);
                return;
            }
        }
        if (HostValues.isHalf(node.in(0))) {
            emitHalf(e, node);
            return;
        }
        FloatArray a = e.f32(node.in(0));
        FloatArray b = e.f32(node.in(1));
        FloatArray c = e.f32(node.out(0));
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        int ta = node.iattr[3];
        int tb = node.iattr[4];
        if (node.iattr.length > 5) {
            int batch = node.iattr[5];
            int strideA = node.iattr[6];
            int strideB = node.iattr[7];
            int tilesY = Emitter.ceilDiv(m, GemmKernels.BM);
            WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, GemmKernels.BN), tilesY * batch, 16, 16);
            FloatArray bias = e.dummyF32();
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64Batched, e.context(), a, b, c, bias, m, n, k, ta, tb, 0, strideA, strideB, tilesY));
            return;
        }
        emitGemm(e, a, b, c, null, m, n, k, ta, tb, 0);
    }

    /** Half-precision operands: the 64x64 register-blocked kernels that convert on load. */
    private static void emitHalf(Emitter e, Node node) {
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        int ta = node.iattr[3];
        int tb = node.iattr[4];
        FloatArray c = e.f32(node.out(0));
        FloatArray bias = e.dummyF32();
        boolean f16 = node.in(0).dtype() == DType.F16;
        if (node.iattr.length > 5 && useTensorCores(e)) {
            emitMma(e, node.in(0), node.in(1), c, bias, m, n, k, ta, tb, 0, node.iattr[5], node.iattr[6], node.iattr[7]);
            return;
        }
        if (node.iattr.length > 5) {
            int batch = node.iattr[5];
            int strideA = node.iattr[6];
            int strideB = node.iattr[7];
            int tilesY = Emitter.ceilDiv(m, GemmKernels.BM);
            WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, GemmKernels.BN), tilesY * batch, 16, 16);
            if (f16) {
                var a = e.f16(node.in(0));
                var b = e.f16(node.in(1));
                e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64BatchedF16, e.context(), a, b, c, bias, m, n, k, ta, tb, 0, strideA, strideB, tilesY));
            } else {
                var a = e.bf16(node.in(0));
                var b = e.bf16(node.in(1));
                e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64BatchedBF16, e.context(), a, b, c, bias, m, n, k, ta, tb, 0, strideA, strideB, tilesY));
            }
            return;
        }
        emitGemmHalf(e, node.in(0), node.in(1), c, bias, m, n, k, ta, tb, 0);
    }

    private static final boolean TENSOR_CORES = !"false".equals(System.getProperty("jitnn.tensorCores"));

    /** True when half-precision products run on tensor cores (CUDA; {@code -Djitnn.tensorCores=false} disables it). */
    static boolean useTensorCores(Emitter e) {
        return TENSOR_CORES && e.isCuda();
    }

    /**
     * Tensor-core GEMM (CUDA, FP32 accumulation) on half-precision operands, batched: C[z] at
     * z * m * n from A[z] at z * strideA and B[z] at z * strideB; batch 1 for a plain product.
     */
    static void emitMma(Emitter e, Tensor at, Tensor bt, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue, int batch, int strideA, int strideB) {
        int tilesY = Emitter.ceilDiv(m, 32);
        WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, 32), tilesY * batch, 32, 1);
        if (at.dtype() == DType.F16) {
            var a = e.f16(at);
            var b = e.f16(bt);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemmMmaF16, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue, strideA, strideB, tilesY));
        } else {
            var a = e.bf16(at);
            var b = e.bf16(bt);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemmMmaBF16, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue, strideA, strideB, tilesY));
        }
    }

    /** C = op(A) op(B) (+ epilogue) for half-precision A and B of the same type, FP32 C. */
    static void emitGemmHalf(Emitter e, Tensor at, Tensor bt, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        if (useTensorCores(e)) {
            if (ta == 0 && tb == 1 && m % MmaGemmKernels.TILE == 0 && n % MmaGemmKernels.TILE == 0 && k % 16 == 0) {
                // jitllm's pipelined 128 x 128 tensor-core GEMM (C = A B^T, whole tiles only)
                WorkerGrid grid = Emitter.blocks2D(m / MmaGemmKernels.TILE, n / MmaGemmKernels.TILE, MmaGemmKernels.THREADS, 1);
                if (at.dtype() == DType.F16) {
                    var a = e.f16(at);
                    var b = e.f16(bt);
                    e.task(grid, (g, id) -> g.task(id, MmaGemmKernels::gemmMMA, e.context(), a, b, c, m, n, k));
                } else {
                    var a = e.bf16(at);
                    var b = e.bf16(bt);
                    e.task(grid, (g, id) -> g.task(id, MmaGemmKernels::gemmMMABF16, e.context(), a, b, c, m, n, k));
                }
                if (epilogue != 0) {
                    e.task(Emitter.grid1D(m * n), (g, id) -> g.task(id, GemmKernels::biasActivation, e.context(), c, bias, m, n, epilogue));
                }
                return;
            }
            emitMma(e, at, bt, c, bias, m, n, k, ta, tb, epilogue, 1, 0, 0);
            return;
        }
        String variant = System.getProperty("jitnn.gemm", "auto");
        boolean big = (long) m * n >= 64L * 64 && k >= 16;
        if ((variant.equals("simdgroup") || (variant.equals("auto") && big)) && e.isMetal() && m % 8 == 0 && n % 8 == 0) {
            // Metal: the simdgroup matrix units, on operands converted to FP32 as they are staged.
            WorkerGrid sg = Emitter.blocks1D(Emitter.ceilDiv(n, 64) * Emitter.ceilDiv(m, 64), 128);
            if (at.dtype() == DType.F16) {
                var a = e.f16(at);
                var b = e.f16(bt);
                e.task(sg, (g, id) -> g.task(id, GemmKernels::gemmSimdgroupF16, e.context(), a, b, c, m, n, k, ta, tb));
            } else {
                var a = e.bf16(at);
                var b = e.bf16(bt);
                e.task(sg, (g, id) -> g.task(id, GemmKernels::gemmSimdgroupBF16, e.context(), a, b, c, m, n, k, ta, tb));
            }
            if (epilogue != 0) {
                e.task(Emitter.grid1D(m * n), (g, id) -> g.task(id, GemmKernels::biasActivation, e.context(), c, bias, m, n, epilogue));
            }
            return;
        }
        WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, GemmKernels.BN), Emitter.ceilDiv(m, GemmKernels.BM), 16, 16);
        if (at.dtype() == DType.F16) {
            var a = e.f16(at);
            var b = e.f16(bt);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64F16, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
        } else {
            var a = e.bf16(at);
            var b = e.bf16(bt);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64BF16, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
        }
    }

    /**
     * Adds C = op(A) op(B) (+ bias, + ReLU per {@code epilogue}, see GemmKernels) to the graph,
     * picking the kernel for the backend and shape: simdgroup matrix units on Metal (the epilogue
     * then runs as a separate kernel), 128x128 register blocks for large outputs, 64x64 blocks
     * otherwise, and the 16x16 tiled kernel for small or thin products. {@code -Djitnn.gemm=}
     * {@code auto|simdgroup|128|64|16} overrides the choice.
     */
    static void emitGemm(Emitter e, FloatArray a, FloatArray b, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        String variant = System.getProperty("jitnn.gemm", "auto");
        boolean auto = variant.equals("auto");
        boolean big = (long) m * n >= 64L * 64 && k >= 16;
        if ((variant.equals("simdgroup") || (auto && big)) && e.isMetal() && m % 8 == 0 && n % 8 == 0) {
            WorkerGrid grid = Emitter.blocks1D(Emitter.ceilDiv(n, 64) * Emitter.ceilDiv(m, 64), 128);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemmSimdgroup, e.context(), a, b, c, m, n, k, ta, tb));
            if (epilogue != 0) {
                FloatArray biasArg = bias != null ? bias : e.dummyF32();
                e.task(Emitter.grid1D(m * n), (g, id) -> g.task(id, GemmKernels::biasActivation, e.context(), c, biasArg, m, n, epilogue));
            }
            return;
        }
        // Every kernel below takes the bias argument; it is only created when one does (J15).
        FloatArray biasArg = bias != null ? bias : e.dummyF32();
        if (variant.equals("128") || (auto && m >= 1024 && n >= 1024 && k >= 64)) {
            WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, 128), Emitter.ceilDiv(m, 128), 16, 16);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm128, e.context(), a, b, c, biasArg, m, n, k, ta, tb, epilogue));
        } else if (variant.equals("64") || (auto && big)) {
            WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, GemmKernels.BN), Emitter.ceilDiv(m, GemmKernels.BM), 16, 16);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64, e.context(), a, b, c, biasArg, m, n, k, ta, tb, epilogue));
        } else {
            e.task(Emitter.grid2D(n, m, GemmKernels.TILE), (g, id) -> g.task(id, GemmKernels::gemm, e.context(), a, b, c, biasArg, m, n, k, ta, tb, epilogue));
        }
    }

    @Override
    public void reference(Node node) {
        Object a = node.in(0).storage().host();
        Object b = node.in(1).storage().host();
        FloatArray c = (FloatArray) node.out(0).storage().host();
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        boolean ta = node.iattr[3] != 0;
        boolean tb = node.iattr[4] != 0;
        int batch = node.iattr.length > 5 ? node.iattr[5] : 1;
        int strideA = node.iattr.length > 5 ? node.iattr[6] : 0;
        int strideB = node.iattr.length > 5 ? node.iattr[7] : 0;
        for (int z = 0; z < batch; z++) {
            int oa = z * strideA;
            int ob = z * strideB;
            int oc = z * m * n;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double acc = 0;
                    for (int q = 0; q < k; q++) {
                        double av = ta ? HostValues.get(a, oa + q * m + i) : HostValues.get(a, oa + i * k + q);
                        double bv = tb ? HostValues.get(b, ob + j * k + q) : HostValues.get(b, ob + q * n + j);
                        acc += av * bv;
                    }
                    c.set(oc + i * n + j, (float) acc);
                }
            }
        }
    }

    @Override
    public Tensor[] backward(Node node, Tensor gcOut) {
        Tensor a = node.in(0);
        Tensor b = node.in(1);
        // Half-precision operands: the gradient products run in that type too (as under PyTorch's autocast).
        Tensor gc = gcOut.to(a.dtype());
        boolean ta = node.iattr[3] != 0;
        boolean tb = node.iattr[4] != 0;
        Tensor ga = null;
        Tensor gb = null;
        boolean needA = a.requiresGrad();
        boolean needB = b.requiresGrad();
        if (!ta && !tb) { // C = A B
            ga = needA ? apply(gc, b, false, true) : null;
            gb = needB ? apply(a, gc, true, false) : null;
        } else if (!ta) { // C = A B^T, B is [N, K]
            ga = needA ? apply(gc, b, false, false) : null;
            gb = needB ? apply(gc, a, true, false) : null;
        } else if (!tb) { // C = A^T B, A is [K, M]
            ga = needA ? apply(b, gc, false, true) : null;
            gb = needB ? apply(a, gc, false, false) : null;
        } else { // C = A^T B^T
            ga = needA ? apply(b, gc, true, true) : null;
            gb = needB ? apply(gc, a, true, true) : null;
        }
        return new Tensor[] { reduceBatch(ga, a), reduceBatch(gb, b) };
    }

    /** The gradient of an input shared by every batch is the sum over the batch. */
    private static Tensor reduceBatch(Tensor g, Tensor input) {
        if (g == null || g.rank() == input.rank()) {
            return g;
        }
        return g.reshape(-1, input.dim(0), input.dim(1)).sum(0, false);
    }
}
