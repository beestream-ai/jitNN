package org.beehive.jitnn.bench;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import org.beehive.jitnn.kernels.MmaGemmKernels;
import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.WorkerGrid3D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Measures FP16 tensor-core GEMMs written with the CUDA Tile API (TileContext) against the
 * KernelContext MMA GEMM jitNN uses today, on the products of BERT-base at 4096 tokens (C = A B^T,
 * B stored [N, K] as a Linear weight). Only with {@code -Djitnn.tileprobe=true}, on CUDA.
 */
class TileGemmProbeAccelTest {

    static final int[][] SHAPES = { { 4096, 2304, 768 }, { 4096, 768, 768 }, { 4096, 3072, 768 }, { 4096, 768, 3072 } };
    static final int RUNS = 30;

    public static void tile128x128x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k, int kBlocks) {
        PartitionView av = tc.partition(tc.view(a, m, k), 128, 32);
        PartitionView bv = tc.partition(tc.view(b, n, k), 128, 32);
        PartitionView cv = tc.partition(tc.view(c, m, n), 128, 128);
        int row = tc.bidX();
        int col = tc.bidY();
        Tile acc = tc.zeros(DType.F32, 128, 128);
        for (int s = 0; s < kBlocks; s++) {
            acc = tc.mma(av.load(row, s), tc.transpose(bv.load(col, s)), acc);
        }
        cv.store(acc, row, col);
    }

    public static void tile128x128x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k, int kBlocks) {
        PartitionView av = tc.partition(tc.view(a, m, k), 128, 64);
        PartitionView bv = tc.partition(tc.view(b, n, k), 128, 64);
        PartitionView cv = tc.partition(tc.view(c, m, n), 128, 128);
        int row = tc.bidX();
        int col = tc.bidY();
        Tile acc = tc.zeros(DType.F32, 128, 128);
        for (int s = 0; s < kBlocks; s++) {
            acc = tc.mma(av.load(row, s), tc.transpose(bv.load(col, s)), acc);
        }
        cv.store(acc, row, col);
    }

    public static void tile64x128x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k, int kBlocks) {
        PartitionView av = tc.partition(tc.view(a, m, k), 64, 32);
        PartitionView bv = tc.partition(tc.view(b, n, k), 128, 32);
        PartitionView cv = tc.partition(tc.view(c, m, n), 64, 128);
        int row = tc.bidX();
        int col = tc.bidY();
        Tile acc = tc.zeros(DType.F32, 64, 128);
        for (int s = 0; s < kBlocks; s++) {
            acc = tc.mma(av.load(row, s), tc.transpose(bv.load(col, s)), acc);
        }
        cv.store(acc, row, col);
    }

    interface Kernel {
        void add(TaskGraph g, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k);
    }

    static HalfFloatArray random(int n, long seed) {
        Random r = new Random(seed);
        HalfFloatArray h = new HalfFloatArray(n);
        for (int i = 0; i < n; i++) {
            h.set(i, new HalfFloat(r.nextFloat() - 0.5f));
        }
        return h;
    }

    /** Median ms of RUNS executions (operands on the device), and the output. */
    static double time(String name, Kernel kernel, WorkerGrid grid, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k) throws Exception {
        // the device under test (jitNN's -Djitnn.device), set before the tasks are added so their sketches are built for it
        String spec = System.getProperty("jitnn.device", "1:0");
        System.setProperty(name + ".device", spec);
        String[] dev = spec.split(":");
        TaskGraph g = new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b);
        kernel.add(g, a, b, c, m, n, k);
        g.transferToHost(DataTransferMode.UNDER_DEMAND, c);
        ImmutableTaskGraph itg = g.snapshot();
        double[] t = new double[RUNS];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            plan.withGridScheduler(new GridScheduler(name + ".k", grid)).withDevice(TornadoExecutionPlan.getDevice(Integer.parseInt(dev[0]), Integer.parseInt(dev[1])));
            for (int i = 0; i < 5; i++) {
                plan.execute();
            }
            for (int i = 0; i < RUNS; i++) {
                long s = System.nanoTime();
                plan.execute();
                t[i] = (System.nanoTime() - s) / 1e6;
            }
            plan.execute().transferToHost(c);
        }
        Arrays.sort(t);
        return t[RUNS / 2];
    }

    @Test
    void probe() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.tileprobe"), "set -Djitnn.tileprobe=true");
        for (int[] s : SHAPES) {
            int m = s[0], n = s[1], k = s[2];
            HalfFloatArray a = random(m * k, 1);
            HalfFloatArray b = random(n * k, 2);
            double gflop = 2.0 * m * n * k / 1e9;
            FloatArray ref = new FloatArray(m * n);
            WorkerGrid mmaGrid = new WorkerGrid3D(m / 128 * MmaGemmKernels.THREADS, n / 128, 1);
            mmaGrid.setLocalWork(MmaGemmKernels.THREADS, 1, 1);
            double tm = time("mma" + n + "x" + k, (g, x, y, z, mm, nn, kk) -> g.task("k", MmaGemmKernels::gemmMMAStrideK, new KernelContext(), x, y, z, mm, nn, kk, kk, 1, kk, 1), mmaGrid, a, b,
                    ref, m, n, k);
            report("KernelContext MMA 128x128x16", m, n, k, tm, gflop, 0);
            Object[][] tiles = { { "tile 128x128x32", 128, 128, 32 }, { "tile 128x128x64", 128, 128, 64 }, { "tile 64x128x32", 64, 128, 32 } };
            for (Object[] v : tiles) {
                String label = (String) v[0];
                int tm0 = (int) v[1], tn0 = (int) v[2], tk0 = (int) v[3];
                FloatArray c = new FloatArray(m * n);
                int kBlocks = k / tk0;
                Kernel kernel = switch (label) {
                    case "tile 128x128x32" -> (g, x, y, z, mm, nn, kk) -> g.task("k", TileGemmProbeAccelTest::tile128x128x32, new TileContext(), x, y, z, mm, nn, kk, kBlocks);
                    case "tile 128x128x64" -> (g, x, y, z, mm, nn, kk) -> g.task("k", TileGemmProbeAccelTest::tile128x128x64, new TileContext(), x, y, z, mm, nn, kk, kBlocks);
                    default -> (g, x, y, z, mm, nn, kk) -> g.task("k", TileGemmProbeAccelTest::tile64x128x32, new TileContext(), x, y, z, mm, nn, kk, kBlocks);
                };
                try {
                    double t = time("t" + tm0 + tn0 + tk0 + "x" + n + "x" + k, kernel, new WorkerGrid2D(m / tm0, n / tn0), a, b, c, m, n, k);
                    double err = 0;
                    for (int i = 0; i < m * n; i += 97) {
                        err = Math.max(err, Math.abs(c.get(i) - ref.get(i)) / (1e-3 + Math.abs(ref.get(i))));
                    }
                    report(label, m, n, k, t, gflop, err);
                } catch (Throwable e) {
                    System.out.printf("[tileprobe] %-30s %5dx%5dx%5d failed: %s%n", label, m, n, k, e);
                }
            }
        }
    }

    static void report(String label, int m, int n, int k, double ms, double gflop, double err) {
        System.out.printf(Locale.ROOT, "[tileprobe] %-30s %5dx%5dx%5d %8.3f ms %7.1f TFLOP/s  max rel err vs MMA %.1e%n", label, m, n, k, ms, gflop / ms, err);
    }
}
