package ai.beestream.jitnn.examples;

import java.util.Arrays;
import java.util.List;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.Cifar10;
import ai.beestream.jitnn.dist.Collective;
import ai.beestream.jitnn.dist.NcclCollective;
import ai.beestream.jitnn.dist.ProcessGroup;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.DataParallel;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.models.ResNet;
import ai.beestream.jitnn.nn.optim.SGD;

/**
 * Trains ResNet-18 on CIFAR-10 across several devices, as PyTorch's DistributedDataParallel
 * examples do: one replica per device, each on 128 images of every step's batch, gradients
 * averaged before each step, so every replica takes the same step. The same training then runs on
 * the first device alone, without any averaging, for the speedup in images per second. Devices are given as TornadoVM lists them ({@code tornado
 * --devices}); {@code --nccl} keeps the averaging on NVIDIA GPUs through NCCL, otherwise it goes
 * through host memory and works with any mix of devices.
 *
 * <pre>run.sh DataParallelCifar10 [--nccl] [device ...]   (default: 0:0 0:1)</pre>
 */
public final class DataParallelCifar10 {

    static final int PER_DEVICE = 128;
    static final int STEPS = 150;

    public static void main(String[] args) throws Exception {
        boolean nccl = Arrays.asList(args).contains("--nccl");
        String[] devices = Arrays.stream(args).filter(a -> !a.startsWith("--")).toArray(String[]::new);
        if (devices.length == 0) {
            devices = new String[] { "0:0", "0:1" };
        }
        Cifar10 train = Cifar10.load(Examples.cifar10(), true);
        Cifar10 test = Cifar10.load(Examples.cifar10(), false);

        double all = run(devices, nccl, train, test);
        Engine.reset();                             // frees the plans and host buffers of the run on all devices
        double one = run(new String[] { devices[0] }, false, train, test);
        System.out.printf("%n%d devices: %.2fx the throughput of one%n", devices.length, one / all);
    }

    /** Trains a fresh model for STEPS steps on the given devices; returns the seconds per step. */
    static double run(String[] devices, boolean nccl, Cifar10 train, Cifar10 test) throws Exception {
        int ranks = devices.length;
        int share = PER_DEVICE;
        System.out.printf("%n%s on %s%s, %d images per device per step%n", ranks == 1 ? "one device" : ranks + " devices", String.join(", ", devices),
                nccl ? " (NCCL)" : "", share);
        long t0 = System.nanoTime();
        List<double[]> results;
        if (nccl) {
            try (NcclCollective c = new NcclCollective(devices)) {
                results = ProcessGroup.run(devices, c, (rank, collective) -> train(rank, collective, share, train, test));
                Engine.reset();                     // the plans go before the communicator
            }
        } else {
            results = ProcessGroup.run(devices, (rank, collective) -> train(rank, collective, share, train, test));
        }
        double[] r0 = results.get(0);
        System.out.printf("  %d steps: %.0f ms per step after the first 10, %.0f images/s; test accuracy %.1f%% (%.0f s in all)%n", STEPS, 1000 * r0[0], ranks * PER_DEVICE / r0[0],
                100 * r0[1], Examples.seconds(t0));
        return r0[0] / ranks;                      // seconds per image, times the per-device batch
    }

    /** One rank: its replica, its share of each batch; returns {seconds per step, test accuracy}. */
    static double[] train(int rank, Collective collective, int share, Cifar10 train, Cifar10 test) {
        Sequential model = ResNet.resnet18(3, 10, true, 42);   // built on this rank's device
        // one device needs no averaging: it trains exactly as Cifar10ResNet does
        DataParallel dp = collective.worldSize() > 1 ? new DataParallel(model, rank, collective).broadcastParameters() : null;
        SGD opt = new SGD(model.parameters(), 0.05f, 0.9f, 5e-4f, false);
        float[] x = new float[share * Cifar10.PIXELS];
        int[] y = new int[share];
        long start = 0;
        for (int step = 0; step < STEPS; step++) {
            if (step == 10) {
                start = System.nanoTime();          // after compiling and warming up
            }
            // the step's batch is the ranks' consecutive blocks of 128 images; this rank takes its own
            train.batch(step * collective.worldSize() + rank, share, x, y);
            Tensor loss = F.crossEntropy(model.forward(Tensor.of(x, share, 3, 32, 32)), Tensor.of(y, share));
            loss.backward();
            if (dp != null) {
                dp.averageGradients();
            }
            opt.step();
            loss.item();
            opt.zeroGrad();
        }
        double perStep = (System.nanoTime() - start) / 1e9 / (STEPS - 10);
        if (rank != 0) {
            return new double[] { perStep, 0 };
        }
        model.train(false);
        int n = 500;
        float[] tx = new float[n * Cifar10.PIXELS];
        int[] ty = new int[n];
        float correct = 0;
        for (int b = 0; b < 4; b++) {
            test.batch(b, n, tx, ty);
            correct += F.accuracy(Grad.inference(() -> model.forward(Tensor.of(tx, n, 3, 32, 32))), Tensor.of(ty, n)) * n;
        }
        return new double[] { perStep, correct / (4 * n) };
    }
}
