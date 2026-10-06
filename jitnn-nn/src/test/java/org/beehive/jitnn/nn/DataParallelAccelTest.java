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
import org.beehive.jitnn.dist.ProcessGroup;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.optim.SGD;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Collectives and data-parallel training across ranks. The ranks run on the devices listed in
 * {@code -Djitnn.dist.devices} (comma-separated "driver:device"; default: two ranks on the test
 * device). Data-parallel training on two half batches must reproduce single-rank training on the
 * full batch.
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
    @SuppressWarnings("unchecked")
    void dataParallelMatchesFullBatch() {
        String[] devices = devices();
        int world = devices.length;
        Object[] single = Engine.onDevice(devices[0], () -> train(0, null, 1));
        List<Object[]> ranks = ProcessGroup.run(devices, (rank, c) -> train(rank, c, world));
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
        System.out.printf("[dist] %d ranks on %s: final loss %.5f, matches full-batch training%n", world, String.join(",", devices), full.get(full.size() - 1));
    }
}
