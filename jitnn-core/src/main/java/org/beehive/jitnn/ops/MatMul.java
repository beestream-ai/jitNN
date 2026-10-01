package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.GemmKernels;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** 2-D matrix product with optional transposes of either operand. iattr = [M, N, K, tA, tB]. */
public final class MatMul implements Op {

    static final MatMul OP = new MatMul();

    private MatMul() {
    }

    public static Tensor apply(Tensor a, Tensor b, boolean transA, boolean transB) {
        Checks.f32(a, b);
        Checks.rank(a, 2);
        Checks.rank(b, 2);
        int m = transA ? a.dim(1) : a.dim(0);
        int ka = transA ? a.dim(0) : a.dim(1);
        int kb = transB ? b.dim(1) : b.dim(0);
        int n = transB ? b.dim(0) : b.dim(1);
        if (ka != kb) {
            throw new IllegalArgumentException("matmul inner dimensions differ: " + a.shape() + (transA ? "^T" : "") + " x " + b.shape() + (transB ? "^T" : ""));
        }
        Tensor out = Tensor.result(Shape.of(m, n), DType.F32);
        Tensor.record(OP, new Tensor[] { a, b }, new Tensor[] { out }, new int[] { m, n, ka, transA ? 1 : 0, transB ? 1 : 0 }, null, true);
        return out;
    }

    @Override
    public String name() {
        return "matmul";
    }

    @Override
    public void emit(Emitter e, Node node) {
        FloatArray a = e.f32(node.in(0));
        FloatArray b = e.f32(node.in(1));
        FloatArray c = e.f32(node.out(0));
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        int ta = node.iattr[3];
        int tb = node.iattr[4];
        emitGemm(e, a, b, c, null, m, n, k, ta, tb, 0);
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
        FloatArray a = (FloatArray) node.in(0).storage().host();
        FloatArray b = (FloatArray) node.in(1).storage().host();
        FloatArray c = (FloatArray) node.out(0).storage().host();
        int m = node.iattr[0];
        int n = node.iattr[1];
        int k = node.iattr[2];
        boolean ta = node.iattr[3] != 0;
        boolean tb = node.iattr[4] != 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0;
                for (int q = 0; q < k; q++) {
                    double av = ta ? a.get(q * m + i) : a.get(i * k + q);
                    double bv = tb ? b.get(j * k + q) : b.get(q * n + j);
                    acc += av * bv;
                }
                c.set(i * n + j, (float) acc);
            }
        }
    }

    @Override
    public Tensor[] backward(Node node, Tensor gc) {
        Tensor a = node.in(0);
        Tensor b = node.in(1);
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
        return new Tensor[] { ga, gb };
    }
}
