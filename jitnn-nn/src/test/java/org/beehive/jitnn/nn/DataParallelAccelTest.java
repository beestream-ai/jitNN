package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Blobs;
import org.beehive.jitnn.dist.Collective;
import org.beehive.jitnn.dist.NcclCollective;
import org.beehive.jitnn.dist.ProcessGroup;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.optim.SGD;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Collectives and data-parallel training across ranks. The ranks run on the devices listed in
 * {@code -Djitnn.dist.devices} (comma-separated "driver:device"; default: two ranks on the test
 * device). Data-parallel training on two half batches must reproduce single-rank training on the
 * full batch. With {@code -Djitnn.dist.nccl=true} the same checks run on {@link NcclCollective}
 * (the devices must then be distinct CUDA GPUs).
 */
class DataParallelAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    static String[] devices() {
        String list = System.getProperty("jitnn.dist.devices", "");
        if (!list.isBlank()) {
            return list.split(",");
        }
        String d = System.getProperty("jitnn.device", "");
        return new String[] { d, d };
    }

    @Test
    void collectives() {
        String[] devices = devices();
        int world = devices.length;
        List<float[][]> out = ProcessGroup.run(devices, (rank, c) -> {
            float[] v = new float[6];
            Arrays.fill(v, rank + 1);
            Tensor x = Tensor.of(v, 2, 3).mul(1.0f);
            c.allReduce(rank, x, Collective.ReduceOp.SUM);
            Tensor y = Tensor.of(new float[] { rank, rank, rank }, 3).mul(1.0f);
            c.broadcast(rank, y, world - 1);
            float[] in = new float[2 * world];
            for (int i = 0; i < in.length; i++) {
                in[i] = i * (rank + 1);
            }
            Tensor rs = Tensor.zeros(2);
            c.reduceScatter(rank, Tensor.of(in, 2 * world).mul(1.0f), rs, Collective.ReduceOp.SUM);
            Tensor ag = Tensor.zeros(world);
            c.allGather(rank, Tensor.of(new float[] { 10.0f * rank }, 1).mul(1.0f), ag);
            return new float[][] { x.toFloatArray(), y.toFloatArray(), rs.toFloatArray(), ag.toFloatArray() };
        });
        float sum = world * (world + 1) / 2.0f;
        float rankSum = sum;
        for (int r = 0; r < world; r++) {
            float[][] o = out.get(r);
            for (float v : o[0]) {
                assertEquals(sum, v, 0.0f, "all-reduce on rank " + r);
            }
            for (float v : o[1]) {
                assertEquals(world - 1, v, 0.0f, "broadcast on rank " + r);
            }
            assertArrayEquals(new float[] { 2 * r * rankSum, (2 * r + 1) * rankSum }, o[2], 1e-4f, "reduce-scatter on rank " + r);
            for (int k = 0; k < world; k++) {
                assertEquals(10.0f * k, o[3][k], 0.0f, "all-gather on rank " + r);
            }
        }
    }

    static Module mlp() {
        return new Sequential(new Linear(8, 32, 11), new ReLU(), new Linear(32, 4, 12));
    }

    /** Losses of 15 SGD steps and the final parameters, training on rows [from, to) of each batch of 64. */
    static Object[] train(int rank, Collective c, int world) {
        Blobs data = new Blobs(640, 8, 4, 1.0f, 3);
        Module model = mlp();
        DataParallel dp = c == null ? null : new DataParallel(model, rank, c).broadcastParameters();
        SGD opt = new SGD(model.parameters(), 0.1f, 0.9f, 0.0f, false);
        int per = 64 / world;
        float[] xb = new float[64 * 8];
        int[] yb = new int[64];
        List<Float> losses = new ArrayList<>();
        for (int step = 0; step < 15; step++) {
            data.batch(step % 10, 64, xb, yb);
            Tensor x = Tensor.of(Arrays.copyOfRange(xb, rank * per * 8, (rank + 1) * per * 8), per, 8);
            Tensor y = Tensor.of(Arrays.copyOfRange(yb, rank * per, (rank + 1) * per), per);
            Tensor loss = F.crossEntropy(model.forward(x), y);
            loss.backward();
            if (dp != null) {
                dp.averageGradients();
            }
            opt.step();
            losses.add(loss.item());
            opt.zeroGrad();
        }
        List<float[]> params = new ArrayList<>();
        for (Tensor p : model.parameters()) {
            params.add(p.toFloatArray());
        }
        return new Object[] { losses, params };
    }

    @Test
    void dataParallelMatchesFullBatch() {
        checkDataParallel(null);
    }

    @Test
    void ncclCollectives() {
        assumeTrue(Boolean.getBoolean("jitnn.dist.nccl"), "no -Djitnn.dist.nccl=true");
        String[] devices = devices();
        int world = devices.length;
        List<float[][]> out;
        try (NcclCollective nccl = new NcclCollective(devices)) {
            out = ProcessGroup.run(devices, nccl, (rank, c) -> {
                // in place on persistent tensors
                float[] v = new float[6];
                Arrays.fill(v, rank + 1);
                Tensor x = Tensor.persistent(Tensor.of(v, 2, 3));
                c.allReduce(rank, x, Collective.ReduceOp.SUM);
                Tensor y = Tensor.persistent(Tensor.of(new float[] { rank, rank, rank }, 3));
                c.broadcast(rank, y, world - 1);
                float[] in = new float[2 * world];
                for (int i = 0; i < in.length; i++) {
                    in[i] = i * (rank + 1);
                }
                Tensor rs = Tensor.persistent(Tensor.zeros(2));
                c.reduceScatter(rank, Tensor.of(in, 2 * world).mul(1.0f), rs, Collective.ReduceOp.SUM);
                Tensor ag = Tensor.persistent(Tensor.zeros(world));
                c.allGather(rank, Tensor.of(new float[] { 10.0f * rank }, 1).mul(1.0f), ag);
                // lazy intermediates, reduced into new tensors through one packed buffer
                Tensor[] m = c.allReduced(rank, new Tensor[] { Tensor.full(rank + 1, 5).mul(2.0f), Tensor.full(-rank, 3, 7).add(1.0f) }, Collective.ReduceOp.AVG);
                c.barrier(rank);
                return new float[][] { x.toFloatArray(), y.toFloatArray(), rs.toFloatArray(), ag.toFloatArray(), m[0].toFloatArray(), m[1].toFloatArray() };
            });
            Engine.reset();
        }
        float sum = world * (world + 1) / 2.0f;
        for (int r = 0; r < world; r++) {
            float[][] o = out.get(r);
            for (float v : o[0]) {
                assertEquals(sum, v, 0.0f, "all-reduce on rank " + r);
            }
            for (float v : o[1]) {
                assertEquals(world - 1, v, 0.0f, "broadcast on rank " + r);
            }
            assertArrayEquals(new float[] { 2 * r * sum, (2 * r + 1) * sum }, o[2], 1e-4f, "reduce-scatter on rank " + r);
            for (int k = 0; k < world; k++) {
                assertEquals(10.0f * k, o[3][k], 0.0f, "all-gather on rank " + r);
            }
            for (float v : o[4]) {
                assertEquals(2.0f * sum / world, v, 1e-5f, "allReduced (first) on rank " + r);
            }
            for (float v : o[5]) {
                assertEquals(1.0f - (sum - world) / world, v, 1e-5f, "allReduced (second) on rank " + r);
            }
        }
    }

    @Test
    void ncclDataParallelMatchesFullBatch() {
        assumeTrue(Boolean.getBoolean("jitnn.dist.nccl"), "no -Djitnn.dist.nccl=true");
        try (NcclCollective nccl = new NcclCollective(devices())) {
            checkDataParallel(nccl);
            Engine.reset();
        }
    }

    /** A small CNN with synchronised batch normalisation: conv, SyncBN, ReLU, pooling, linear. */
    static Module syncCnn(Collective c, int rank) {
        return new Sequential(new Conv2d(3, 8, 3, 1, 1, 21), new SyncBatchNorm2d(8, c, rank), new ReLU(), new GlobalAvgPool2d(), new Linear(8, 4, 22));
    }

    /** Parameters and running statistics after 6 SGD steps on rows [rank * per, (rank + 1) * per) of each batch of 16. */
    static List<float[]> trainSync(int rank, Collective c, int world) {
        Module model = syncCnn(world == 1 ? null : c, rank);
        DataParallel dp = c == null ? null : new DataParallel(model, rank, c).broadcastParameters();
        SGD opt = new SGD(model.parameters(), 0.05f, 0.9f, 0.0f, false);
        java.util.Random rnd = new java.util.Random(5);
        int per = 16 / world;
        for (int step = 0; step < 6; step++) {
            float[] xb = new float[16 * 3 * 6 * 6];
            int[] yb = new int[16];
            for (int i = 0; i < xb.length; i++) {
                xb[i] = rnd.nextFloat() * 2 - 1 + (i % 3);
            }
            for (int i = 0; i < 16; i++) {
                yb[i] = rnd.nextInt(4);
            }
            Tensor x = Tensor.of(Arrays.copyOfRange(xb, rank * per * 108, (rank + 1) * per * 108), per, 3, 6, 6);
            Tensor y = Tensor.of(Arrays.copyOfRange(yb, rank * per, (rank + 1) * per), per);
            Tensor loss = F.crossEntropy(model.forward(x), y);
            loss.backward();
            if (dp != null) {
                dp.averageGradients();
            }
            opt.step();
            opt.zeroGrad();
        }
        List<float[]> out = new ArrayList<>();
        for (Tensor p : model.parameters()) {
            out.add(p.toFloatArray());
        }
        for (Tensor b : model.buffers()) {
            out.add(b.toFloatArray());
        }
        return out;
    }

    @Test
    void ncclSyncBatchNormMatchesFullBatch() {
        assumeTrue(Boolean.getBoolean("jitnn.dist.nccl"), "no -Djitnn.dist.nccl=true");
        String[] devices = devices();
        int world = devices.length;
        List<float[]> single = Engine.onDevice(devices[0], () -> trainSync(0, null, 1));
        List<List<float[]>> ranks;
        try (NcclCollective nccl = new NcclCollective(devices)) {
            ranks = ProcessGroup.run(devices, nccl, (rank, c) -> trainSync(rank, c, world));
            Engine.reset();
        }
        for (int r = 0; r < world; r++) {
            for (int i = 0; i < single.size(); i++) {
                float worst = 0;
                for (int k = 0; k < single.get(i).length; k++) {
                    worst = Math.max(worst, Math.abs(single.get(i)[k] - ranks.get(r).get(i)[k]));
                }
                assertTrue(worst <= 1e-4f, "rank " + r + " tensor " + i + " differs from full-batch training by " + worst);
            }
        }
        System.out.printf("[dist] SyncBatchNorm2d on %d ranks matches batch normalisation of the full batch%n", world);
    }

    @SuppressWarnings("unchecked")
    private static void checkDataParallel(Collective collective) {
        String[] devices = devices();
        int world = devices.length;
        Object[] single = Engine.onDevice(devices[0], () -> train(0, null, 1));
        List<Object[]> ranks = collective == null ? ProcessGroup.run(devices, (rank, c) -> train(rank, c, world))
                : ProcessGroup.run(devices, collective, (rank, c) -> train(rank, c, world));
        List<float[]> ref = (List<float[]>) single[1];
        for (int r = 0; r < world; r++) {
            List<float[]> p = (List<float[]>) ranks.get(r)[1];
            for (int i = 0; i < ref.size(); i++) {
                float worst = 0;
                for (int k = 0; k < ref.get(i).length; k++) {
                    worst = Math.max(worst, Math.abs(ref.get(i)[k] - p.get(i)[k]));
                }
                assertTrue(worst <= 2e-5f, "rank " + r + " parameter " + i + " differs from full-batch training by " + worst);
            }
        }
        // the full-batch loss is the mean of the ranks' half-batch losses
        List<Float> full = (List<Float>) single[0];
        for (int s = 0; s < full.size(); s++) {
            float mean = 0;
            for (int r = 0; r < world; r++) {
                mean += ((List<Float>) ranks.get(r)[0]).get(s) / world;
            }
            assertEquals(full.get(s), mean, 2e-5f * Math.max(1, Math.abs(full.get(s))), "loss at step " + s);
        }
        System.out.printf("[dist] %d ranks on %s (%s): final loss %.5f, matches full-batch training%n", world, String.join(",", devices),
                collective == null ? "host" : "nccl", full.get(full.size() - 1));
    }
}
