package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Autotune;
import org.beehive.jitnn.engine.Tuning;
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

    /** Work-groups a forward product should reach before splitting K stops (about two per SM on an A100). */
    private static final int SPLIT_TARGET = Integer.getInteger("jitnn.splitK.groups", 256);

    /**
     * K slices for a forward product C = op(A) op(B): 1 when its 64 x 64 output tiles already fill
     * the GPU, otherwise doubled while the work-groups stay under {@link #SPLIT_TARGET} and each
     * slice keeps at least 64 of K.
     */
    static int forwardSlices(int m, int n, int k) {
        if ((long) m * n < 64L * 64) {
            return 1;
        }
        return forwardSlices(Emitter.ceilDiv(m, GemmKernels.BM) * Emitter.ceilDiv(n, GemmKernels.BN), k);
    }

    /** As {@link #forwardSlices(int, int, int)} for {@code tiles} work-groups over a reduction of {@code k}. */
    static int forwardSlices(int tiles, int k) {
        if (!SPLIT_K) {
            return 1;
        }
        int slices = 1;
        while (tiles * slices * 2 <= SPLIT_TARGET && k / (slices * 2) >= 64) {
            slices *= 2;
        }
        return slices;
    }

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
        if (dt != DType.F32 && useTensorCores(e) && wholeTiles(m, n, ks)) {
            emitMmaTiles(e, node.in(0), node.in(1), part, m, n, k, 1, 0, slices);
        } else if (dt != DType.F32 && useTensorCores(e)) {
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
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64pdBatched, e.context(), a, b, part, bias, m, n, ks, 1, 0, 0, strideA, strideB, tilesY));
        }
        int cells = m * n;
        e.task(Emitter.grid1D(cells), (g, id) -> g.task(id, org.beehive.jitnn.kernels.NormKernels::sumPartials, e.context(), part, c, slices, cells));
    }

    @Override
    public void emit(Emitter e, Node node) {
        if (emitNative(e, node)) {
            return;
        }
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
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64pdBatched, e.context(), a, b, c, bias, m, n, k, ta, tb, 0, strideA, strideB, tilesY));
            return;
        }
        emitGemm(e, a, b, c, null, m, n, k, ta, tb, 0);
    }

    /** The product as a vendor-library call (cuBLAS, MLX) when {@code -Djitnn.native} enables one: false otherwise. */
    private static boolean emitNative(Emitter e, Node node) {
        int[] a = node.iattr;
        FloatArray c = e.f32(node.out(0));
        if (HostValues.isHalf(node.in(0))) {
            if (a.length == 5 && node.in(0).dtype() == DType.BF16 && node.in(1).dtype() == DType.BF16) {
                return org.beehive.jitnn.natives.Libs.gemmBF16(e, e.bf16(node.in(0)), e.bf16(node.in(1)), c, null, a[0], a[1], a[2], a[3], a[4], 0);
            }
            return a.length == 5 && node.in(0).dtype() == DType.F16 && node.in(1).dtype() == DType.F16
                    && org.beehive.jitnn.natives.Libs.gemmF16(e, e.f16(node.in(0)), e.f16(node.in(1)), c, null, a[0], a[1], a[2], a[3], a[4], 0);
        }
        if (a.length > 5) {
            return org.beehive.jitnn.natives.Libs.gemmBatched(e, e.f32(node.in(0)), e.f32(node.in(1)), c, null, a[0], a[1], a[2], a[3], a[4], 0, a[5], a[6], a[7]);
        }
        return org.beehive.jitnn.natives.Libs.gemm(e, e.f32(node.in(0)), e.f32(node.in(1)), c, null, a[0], a[1], a[2], a[3], a[4], 0);
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
     * K slices for a forward tensor-core product on whole 128 x 128 tiles: doubled while the
     * work-groups stay under the split-K target and each slice is a multiple of 16, at least 64 deep.
     */
    static int mmaSlices(int m, int n, int k) {
        if (!SPLIT_K) {
            return 1;
        }
        int tiles = (m / MmaGemmKernels.TILE) * (n / MmaGemmKernels.TILE);
        int slices = 1;
        while (tiles * slices * 2 <= SPLIT_TARGET && k % (slices * 2 * 16) == 0 && k / (slices * 2) >= 64) {
            slices *= 2;
        }
        return slices;
    }

    /** Whether the tensor-core kernels are enabled ({@code -Djitnn.tensorCores=false} turns them off). */
    static boolean tensorCores() {
        return TENSOR_CORES;
    }

    /**
     * C = A B^T (+ epilogue) for an FP32 A [m, k] and an FP16 B [n, k] on the tensor cores, A rounded
     * to FP16 as it is staged; whole 128 x 128 tiles ({@link #wholeTiles}). Split along K like
     * {@link #emitGemmHalf}.
     */
    static void emitGemmA32(Emitter e, FloatArray a, Tensor bt, FloatArray c, FloatArray bias, int m, int n, int k, int epilogue) {
        var b = e.f16(bt);
        if (org.beehive.jitnn.natives.Libs.gemmA32(e, a, b, c, bias, m, n, k, epilogue)) {
            return;
        }
        int slices = mmaSlices(m, n, k);
        int ks = k / slices;
        WorkerGrid grid = Emitter.blocks3D(m / MmaGemmKernels.TILE, n / MmaGemmKernels.TILE, slices, MmaGemmKernels.THREADS);
        FloatArray out = slices > 1 ? e.scratch(slices * m * n) : c;
        e.task(grid, (g, id) -> g.task(id, MmaGemmKernels::gemmMMAStrideKA32, e.context(), a, b, out, m, n, ks, k, 1, k, 1));
        int mn = m * n;
        if (slices > 1) {
            e.task(Emitter.grid1D(mn), (g, id) -> g.task(id, GemmKernels::sumSplitK, e.context(), out, c, bias, slices, mn, n, epilogue));
        } else if (epilogue != 0) {
            e.task(Emitter.grid1D(mn), (g, id) -> g.task(id, GemmKernels::biasActivation, e.context(), c, bias, m, n, epilogue));
        }
    }

    /** True when the 128 x 128 tensor-core kernels cover the product (whole tiles, K in steps of 16). */
    static boolean wholeTiles(int m, int n, int k) {
        return m % MmaGemmKernels.TILE == 0 && n % MmaGemmKernels.TILE == 0 && k % 16 == 0;
    }

    /**
     * C = op(A) op(B) with jitllm's 128 x 128 tensor-core kernel generalised to strided operands;
     * with {@code slices > 1}, c holds the slices x M x N partial products of a split along K.
     */
    static void emitMmaTiles(Emitter e, Tensor at, Tensor bt, FloatArray c, int m, int n, int k, int ta, int tb, int slices) {
        int ks = k / slices;
        int sAr = ta == 0 ? k : 1;
        int sAk = ta == 0 ? 1 : m;
        int sBc = tb == 1 ? k : 1;
        int sBk = tb == 1 ? 1 : n;
        WorkerGrid grid = Emitter.blocks3D(m / MmaGemmKernels.TILE, n / MmaGemmKernels.TILE, slices, MmaGemmKernels.THREADS);
        if (at.dtype() == DType.F16) {
            var a = e.f16(at);
            var b = e.f16(bt);
            e.task(grid, (g, id) -> g.task(id, MmaGemmKernels::gemmMMAStrideK, e.context(), a, b, c, m, n, ks, sAr, sAk, sBc, sBk));
        } else {
            var a = e.bf16(at);
            var b = e.bf16(bt);
            e.task(grid, (g, id) -> g.task(id, MmaGemmKernels::gemmMMAStrideKBF16, e.context(), a, b, c, m, n, ks, sAr, sAk, sBc, sBk));
        }
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
        if (at.dtype() == DType.F16 && bt.dtype() == DType.F16 && org.beehive.jitnn.natives.Libs.gemmF16(e, e.f16(at), e.f16(bt), c, bias, m, n, k, ta, tb, epilogue)) {
            return;
        }
        if (at.dtype() == DType.BF16 && bt.dtype() == DType.BF16 && org.beehive.jitnn.natives.Libs.gemmBF16(e, e.bf16(at), e.bf16(bt), c, bias, m, n, k, ta, tb, epilogue)) {
            return;
        }
        if (useTensorCores(e)) {
            if (wholeTiles(m, n, k)) {
                int slices = mmaSlices(m, n, k);
                if (slices > 1) {
                    // too few 128 x 128 tiles to fill the GPU: K slices, summed with the epilogue
                    FloatArray part = e.scratch(slices * m * n);
                    emitMmaTiles(e, at, bt, part, m, n, k, ta, tb, slices);
                    int mn = m * n;
                    e.task(Emitter.grid1D(mn), (g, id) -> g.task(id, GemmKernels::sumSplitK, e.context(), part, c, bias, slices, mn, n, epilogue));
                    return;
                }
                // jitllm's pipelined 128 x 128 tensor-core GEMM, on strided operands (any transpose)
                emitMmaTiles(e, at, bt, c, m, n, k, ta, tb, 1);
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
        if (org.beehive.jitnn.natives.Libs.gemm(e, a, b, c, bias, m, n, k, ta, tb, epilogue)) {
            return;
        }
        String variant = System.getProperty("jitnn.gemm", "auto");
        boolean big = (long) m * n >= 64L * 64 && k >= 16;
        if (variant.equals("auto") && !e.isMetal()) {
            // the device's tuning table, measured offline per shape class (Tuning), else the rules
            // below; with -Djitnn.autotune the variants are measured on this device (Autotune)
            String tuned = Tuning.gemm(m, n, k, ta, tb);
            int forward = forwardSlices(m, n, k);
            String rule = tuned != null ? tuned : forward > 1 ? "splitK" : big ? "gemm64pd" : "gemm16";
            if (Autotune.enabled()) {
                java.util.List<String> candidates = new java.util.ArrayList<>(java.util.List.of("gemm64pd", "gemm64p", "gemm128pd", "gemm64x128pd", "gemm128x64pd", "gemm16"));
                if (forward > 1) {
                    candidates.add("splitK");
                }
                if (!candidates.contains(rule)) {
                    candidates.add(rule);
                }
                variant = Autotune.choose(e, "gemm", Tuning.gemmShape(m, n, k, ta, tb), rule, candidates.toArray(new String[0]));
            } else if (tuned != null) {
                variant = tuned;
            }
        }
        boolean auto = variant.equals("auto");
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
        int slices = auto || variant.equals("splitK") ? forwardSlices(m, n, k) : 1;
        if (slices > 1) {
            // too few output tiles to fill the GPU: split K, then sum the partials with the epilogue
            int ks = Emitter.ceilDiv(Emitter.ceilDiv(k, slices), 16) * 16;
            int sl = Emitter.ceilDiv(k, ks);
            int tilesY = Emitter.ceilDiv(m, GemmKernels.BM);
            FloatArray part = e.scratch(sl * m * n);
            e.task(Emitter.blocks2D(Emitter.ceilDiv(n, GemmKernels.BN), tilesY * sl, 16, 16),
                    (g, id) -> g.task(id, GemmKernels::gemm64SplitK, e.context(), a, b, part, m, n, k, ta, tb, ks, tilesY));
            int mn = m * n;
            e.task(Emitter.grid1D(mn), (g, id) -> g.task(id, GemmKernels::sumSplitK, e.context(), part, c, biasArg, sl, mn, n, epilogue));
            return;
        }
        if (tuned(e, variant, a, b, c, biasArg, m, n, k, ta, tb, epilogue)) {
            return;
        }
        if (auto && big) {
            // padded local memory and register prefetch: 1.5x the earlier 64 and 128 kernels over the
            // reference workloads' products on an A100 (bench/results/a100, GEMM sweep)
            tuned(e, "gemm64pd", a, b, c, biasArg, m, n, k, ta, tb, epilogue);
        } else if (variant.equals("128")) {
            WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, 128), Emitter.ceilDiv(m, 128), 16, 16);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm128, e.context(), a, b, c, biasArg, m, n, k, ta, tb, epilogue));
        } else if (variant.equals("64") || (auto && big)) {
            WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, GemmKernels.BN), Emitter.ceilDiv(m, GemmKernels.BM), 16, 16);
            e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64, e.context(), a, b, c, biasArg, m, n, k, ta, tb, epilogue));
        } else {
            e.task(Emitter.grid2D(n, m, GemmKernels.TILE), (g, id) -> g.task(id, GemmKernels::gemm, e.context(), a, b, c, biasArg, m, n, k, ta, tb, epilogue));
        }
    }

    /**
     * The FP32 GEMM variants that pad local memory and prefetch the next tile (scripts/codegen/gemm.py
     * pad, dbuf), chosen by name with {@code -Djitnn.gemm=}: returns false for any other name.
     */
    static boolean tuned(Emitter e, String variant, FloatArray a, FloatArray b, FloatArray c, FloatArray bias, int m, int n, int k, int ta, int tb, int epilogue) {
        int bm;
        int bn;
        switch (variant) {
            case "gemm64p", "gemm64pd" -> {
                bm = 64;
                bn = 64;
            }
            case "gemm128p", "gemm128pd" -> {
                bm = 128;
                bn = 128;
            }
            case "gemm128x64pd" -> {
                bm = 128;
                bn = 64;
            }
            case "gemm64x128pd" -> {
                bm = 64;
                bn = 128;
            }
            default -> {
                return false;
            }
        }
        WorkerGrid grid = Emitter.blocks2D(Emitter.ceilDiv(n, bn), Emitter.ceilDiv(m, bm), 16, 16);
        switch (variant) {
            case "gemm64p" -> e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64p, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
            case "gemm64pd" -> e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64pd, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
            case "gemm128p" -> e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm128p, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
            case "gemm128pd" -> e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm128pd, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
            case "gemm128x64pd" -> e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm128x64pd, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
            default -> e.task(grid, (g, id) -> g.task(id, GemmKernels::gemm64x128pd, e.context(), a, b, c, bias, m, n, k, ta, tb, epilogue));
        }
        return true;
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
