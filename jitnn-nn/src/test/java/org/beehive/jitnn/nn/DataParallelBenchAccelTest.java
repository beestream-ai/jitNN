package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.dist.Collective;
import org.beehive.jitnn.dist.HostCollective;
import org.beehive.jitnn.dist.NcclCollective;
import org.beehive.jitnn.dist.ProcessGroup;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.models.ResNet;
import org.beehive.jitnn.nn.optim.SGD;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Data-parallel training throughput: one device against the ranks of {@code -Djitnn.dist.devices}
 * with the host collective and with NCCL, each rank on the same per-device batch. Runs with
 * {@code -Djitnn.bench.dist=true}; {@code -Djitnn.bench.dist.model=resnet18} (CIFAR-size, batch
 * 128 per device, default) or {@code resnet50} (224 x 224, batch 32), {@code -Djitnn.bench.dist.batch},
 * {@code -Djitnn.bench.dist.steps} (30, after 5 warm-up steps), {@code -Djitnn.bench.dist.modes}
 * (a subset of {@code one,host,nccl}, to run them in separate JVMs).
 */
class DataParallelBenchAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    private static final String MODEL = System.getProperty("jitnn.bench.dist.model", "resnet18");
    private static final boolean BIG = MODEL.equals("resnet50");
    private static final int SIZE = BIG ? 224 : 32;
    private static final int BATCH = Integer.getInteger("jitnn.bench.dist.batch", BIG ? 32 : 128);
    private static final int STEPS = Integer.getInteger("jitnn.bench.dist.steps", 30);
    private static final int WARMUP = 5;
    private static final int CLASSES = BIG ? 1000 : 10;
    private static final java.util.Set<String> MODES = java.util.Set.of(System.getProperty("jitnn.bench.dist.modes", "one,host,nccl").split(","));

    /** Median step time in ms of rank {@code rank}'s training loop. */
    static double train(int rank, Collective c) {
        Module model = BIG ? ResNet.resnet50(3, CLASSES, false, 7) : ResNet.resnet18(3, CLASSES, true, 7);
        DataParallel dp = c == null ? null : new DataParallel(model, rank, c).broadcastParameters();
        SGD opt = new SGD(model.parameters(), 0.01f, 0.9f, 0.0f, false);
        Random rnd = new Random(rank);
        float[] x = new float[BATCH * 3 * SIZE * SIZE];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() - 0.5f;
        }
        int[] y = new int[BATCH];
        for (int i = 0; i < BATCH; i++) {
            y[i] = rnd.nextInt(CLASSES);
        }
        List<Double> times = new ArrayList<>();
        long[] before = null;
        for (int step = 0; step < WARMUP + STEPS; step++) {
            if (step == WARMUP) {
                before = Engine.trafficStats();
                if (Boolean.getBoolean("jitnn.profile")) {
                    Engine.kernelBreakdown();
                }
            }
            long t0 = System.nanoTime();
            Tensor loss = F.crossEntropy(model.forward(Tensor.wrap(x, BATCH, 3, SIZE, SIZE)), Tensor.wrap(y, BATCH));
            loss.backward();
            if (dp != null) {
                dp.averageGradients();
            }
            opt.step();
            opt.zeroGrad();
            float l = loss.item();
            if (!Float.isFinite(l)) {
                throw new IllegalStateException("loss " + l + " at step " + step);
            }
            if (step >= WARMUP) {
                times.add((System.nanoTime() - t0) / 1e6);
            }
        }
        if (c == null && Boolean.getBoolean("jitnn.profile")) {
            // device time per operator task over the timed steps, largest first
            java.util.Map<String, long[]> k = Engine.kernelBreakdown();
            k.entrySet().stream().sorted((p1, p2) -> Long.compare(p2.getValue()[1], p1.getValue()[1])).limit(25)
                    .forEach(en -> System.out.printf("[profile] %-40s %8.3f ms per step, %d launches%n", en.getKey(), en.getValue()[1] / 1e6 / STEPS, en.getValue()[0] / STEPS));
        }
        if (c == null) {
            long[] after = Engine.trafficStats();
            System.out.printf("[traffic] %s training step, batch %d: %.0f kernels %.0f library calls %.1f MB%n", MODEL, BATCH, (after[1] - before[1]) / (double) STEPS,
                    (after[2] - before[2]) / (double) STEPS, (after[3] - before[3]) / 1e6 / STEPS);
        }
        double[] t = times.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return t[t.length / 2];
    }

    @Test
    void throughput() {
        assumeTrue(Boolean.getBoolean("jitnn.bench.dist"), "no -Djitnn.bench.dist=true");
        // the one-device step alone runs on the test device too (a training benchmark)
        String[] devices = System.getProperty("jitnn.dist.devices", System.getProperty("jitnn.test.device", "")).split(",");
        int world = devices.length;
        boolean ranks = MODES.contains("host") || MODES.contains("nccl");
        assumeTrue(!devices[0].isEmpty() && (!ranks || world >= 2), "needs -Djitnn.dist.devices with two or more devices");
        double one = Double.NaN;
        if (MODES.contains("one")) {
            one = Engine.onDevice(devices[0], () -> train(0, null));
            report("1 device", 1, one);
            Engine.reset();
        }
        if (MODES.contains("host")) {
            List<Double> host = ProcessGroup.run(devices, new HostCollective(world), DataParallelBenchAccelTest::train);
            report(world + " devices, host collective", world, max(host));
            Engine.reset();
        }
        if (MODES.contains("nccl")) {
            try (NcclCollective nccl = new NcclCollective(devices)) {
                List<Double> n = ProcessGroup.run(devices, nccl, DataParallelBenchAccelTest::train);
                report(world + " devices, NCCL", world, max(n));
                if (!Double.isNaN(one)) {
                    System.out.printf("[dist-bench] NCCL scaling: %.2fx the samples/s of 1 device%n", world * one / max(n));
                }
                Engine.reset();
            }
        }
    }

    private static double max(List<Double> v) {
        return v.stream().mapToDouble(Double::doubleValue).max().orElseThrow();
    }

    private static void report(String what, int world, double ms) {
        System.out.printf("[dist-bench] %s %s batch %d per device: step %.2f ms, %.0f samples/s%n", MODEL, what, BATCH, ms, world * BATCH / (ms / 1e3));
    }
}
