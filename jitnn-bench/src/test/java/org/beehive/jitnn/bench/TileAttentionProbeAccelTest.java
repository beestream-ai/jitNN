package org.beehive.jitnn.bench;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Measures flash attention written with the CUDA Tile API on BERT-base's attention at batch 32
 * (12 heads, 128 tokens, head dimension 64): FP32 packed QKV [B, T, 3, H, d] read through strided
 * views, tiles rounded to FP16 for the tensor-core products, FP32 output [B, T, H, d]. Only with
 * {@code -Djitnn.tileprobe=true}, on CUDA.
 */
class TileAttentionProbeAccelTest {

    static final int B = 32, T = 128, H = 12, D = 64;
    static final float LIMIT = -1.0e30f;

    public static void flash64(TileContext tc, FloatArray qkv, FloatArray out, int heads, int t, int kvBlocks, float scale) {
        int z = tc.bidY();
        int b = z / heads;
        int h = z - b * heads;
        int rowStride = 3 * heads * 64;
        int base = b * t * rowStride + h * 64;
        PartitionView qView = tc.partition(tc.viewStrided(qkv, base, t, 64, rowStride), 64, 64);
        PartitionView kView = tc.partition(tc.viewStrided(qkv, base + heads * 64, t, 64, rowStride), 64, 64);
        PartitionView vView = tc.partition(tc.viewStrided(qkv, base + 2 * heads * 64, t, 64, rowStride), 64, 64);
        PartitionView oView = tc.partition(tc.viewStrided(out, b * t * heads * 64 + h * 64, t, 64, heads * 64), 64, 64);
        int qb = tc.bidX();
        Tile query = tc.cast(qView.load(qb, 0), DType.F16);
        Tile rowMax = tc.full(DType.F32, LIMIT, 64, 1);
        Tile rowSum = tc.zeros(DType.F32, 64, 1);
        Tile acc = tc.zeros(DType.F32, 64, 64);
        for (int blk = 0; blk < kvBlocks; blk++) {
            Tile keys = tc.transpose(tc.cast(kView.load(blk, 0), DType.F16));
            Tile scores = tc.scale(tc.mma(query, keys, tc.zeros(DType.F32, 64, 64)), scale);
            Tile newMax = tc.maximum(rowMax, tc.max(scores, 1));
            Tile p = tc.exp(tc.sub(scores, newMax));
            Tile corr = tc.exp(tc.sub(rowMax, newMax));
            rowSum = tc.add(tc.mul(rowSum, corr), tc.sum(p, 1));
            acc = tc.mul(acc, corr);
            acc = tc.mma(tc.cast(p, DType.F16), tc.cast(vView.load(blk, 0), DType.F16), acc);
            rowMax = newMax;
        }
        oView.store(tc.div(acc, rowSum), qb, 0);
    }

    public static void flash128(TileContext tc, FloatArray qkv, FloatArray out, int heads, int t, int kvBlocks, float scale) {
        int z = tc.bidY();
        int b = z / heads;
        int h = z - b * heads;
        int rowStride = 3 * heads * 64;
        int base = b * t * rowStride + h * 64;
        PartitionView qView = tc.partition(tc.viewStrided(qkv, base, t, 64, rowStride), 128, 64);
        PartitionView kView = tc.partition(tc.viewStrided(qkv, base + heads * 64, t, 64, rowStride), 64, 64);
        PartitionView vView = tc.partition(tc.viewStrided(qkv, base + 2 * heads * 64, t, 64, rowStride), 64, 64);
        PartitionView oView = tc.partition(tc.viewStrided(out, b * t * heads * 64 + h * 64, t, 64, heads * 64), 128, 64);
        int qb = tc.bidX();
        Tile query = tc.cast(qView.load(qb, 0), DType.F16);
        Tile rowMax = tc.full(DType.F32, LIMIT, 128, 1);
        Tile rowSum = tc.zeros(DType.F32, 128, 1);
        Tile acc = tc.zeros(DType.F32, 128, 64);
        for (int blk = 0; blk < kvBlocks; blk++) {
            Tile keys = tc.transpose(tc.cast(kView.load(blk, 0), DType.F16));
            Tile scores = tc.scale(tc.mma(query, keys, tc.zeros(DType.F32, 128, 64)), scale);
            Tile newMax = tc.maximum(rowMax, tc.max(scores, 1));
            Tile p = tc.exp(tc.sub(scores, newMax));
            Tile corr = tc.exp(tc.sub(rowMax, newMax));
            rowSum = tc.add(tc.mul(rowSum, corr), tc.sum(p, 1));
            acc = tc.mul(acc, corr);
            acc = tc.mma(tc.cast(p, DType.F16), tc.cast(vView.load(blk, 0), DType.F16), acc);
            rowMax = newMax;
        }
        oView.store(tc.div(acc, rowSum), qb, 0);
    }

    @Test
    void probe() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.tileprobe"), "set -Djitnn.tileprobe=true");
        FloatArray qkv = new FloatArray(B * T * 3 * H * D);
        Random r = new Random(3);
        for (int i = 0; i < qkv.getSize(); i++) {
            qkv.set(i, r.nextFloat() * 2 - 1);
        }
        float scale = (float) (1.0 / Math.sqrt(D));
        double gflop = 4.0 * B * H * T * T * D / 1e9;
        // host reference for a few (z, row) pairs
        for (String v : new String[] { "flash64", "flash128" }) {
            int bm = v.equals("flash64") ? 64 : 128;
            FloatArray out = new FloatArray(B * T * H * D);
            String name = "attn" + bm;
            String spec = System.getProperty("jitnn.device", "1:0");
            System.setProperty(name + ".device", spec);
            String[] dev = spec.split(":");
            TaskGraph g = new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, qkv);
            if (bm == 64) {
                g.task("k", TileAttentionProbeAccelTest::flash64, new TileContext(), qkv, out, H, T, T / 64, scale);
            } else {
                g.task("k", TileAttentionProbeAccelTest::flash128, new TileContext(), qkv, out, H, T, T / 64, scale);
            }
            g.transferToHost(DataTransferMode.UNDER_DEMAND, out);
            double[] t = new double[30];
            try {
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                    plan.withGridScheduler(new GridScheduler(name + ".k", new WorkerGrid2D(T / bm, B * H)))
                            .withDevice(TornadoExecutionPlan.getDevice(Integer.parseInt(dev[0]), Integer.parseInt(dev[1])));
                    for (int i = 0; i < 5; i++) {
                        plan.execute();
                    }
                    for (int i = 0; i < t.length; i++) {
                        long s = System.nanoTime();
                        plan.execute();
                        t[i] = (System.nanoTime() - s) / 1e6;
                    }
                    plan.execute().transferToHost(out);
                }
            } catch (Throwable e) {
                System.out.printf("[tileprobe] attention %s failed: %s%n", v, e);
                continue;
            }
            Arrays.sort(t);
            double err = 0;
            for (int z = 0; z < B * H; z += 37) {
                int b = z / H, h = z % H;
                for (int i = 0; i < T; i += 13) {
                    double[] sc = new double[T];
                    double mx = -1e300;
                    for (int j = 0; j < T; j++) {
                        double dot = 0;
                        for (int d = 0; d < D; d++) {
                            dot += qkv.get(((b * T + i) * 3) * H * D + h * D + d) * qkv.get(((b * T + j) * 3 + 1) * H * D + h * D + d);
                        }
                        sc[j] = dot * scale;
                        mx = Math.max(mx, sc[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < T; j++) {
                        sc[j] = Math.exp(sc[j] - mx);
                        sum += sc[j];
                    }
                    for (int d = 0; d < D; d++) {
                        double o = 0;
                        for (int j = 0; j < T; j++) {
                            o += sc[j] / sum * qkv.get(((b * T + j) * 3 + 2) * H * D + h * D + d);
                        }
                        err = Math.max(err, Math.abs(o - out.get(((b * T + i) * H + h) * D + d)));
                    }
                }
            }
            System.out.printf(Locale.ROOT, "[tileprobe] attention %-8s %d x %d heads, T=S=%d, d=%d: %8.3f ms  %6.1f TFLOP/s  max abs err %.1e (BERT FP32 flash: 0.221 ms)%n", v, B, H, T, D, t[15],
                    gflop / t[15], err);
        }
    }
}
