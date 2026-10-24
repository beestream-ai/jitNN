package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Blobs;
import org.beehive.jitnn.io.StateDict;
import org.beehive.jitnn.nn.optim.AdaDelta;
import org.beehive.jitnn.nn.optim.AdaGrad;
import org.beehive.jitnn.nn.optim.AdaMax;
import org.beehive.jitnn.nn.optim.Adam;
import org.beehive.jitnn.nn.optim.LRScheduler;
import org.beehive.jitnn.nn.optim.Lamb;
import org.beehive.jitnn.nn.optim.Lars;
import org.beehive.jitnn.nn.optim.NAdam;
import org.beehive.jitnn.nn.optim.Optimizer;
import org.beehive.jitnn.nn.optim.RMSProp;
import org.beehive.jitnn.nn.optim.SGD;

/** Checkpoint checks shared by the reference and device tests. */
final class CheckpointRun {

    private static final int DIMS = 16;
    private static final int CLASSES = 3;
    private static final int BATCH = 64;
    private static final Blobs DATA = new Blobs(512, DIMS, CLASSES, 0.8f, 11);

    private CheckpointRun() {
    }

    static List<Function<List<Tensor>, Optimizer>> optimizers() {
        return List.of(
                p -> new Adam(p, 0.01f),
                p -> new Adam(p, 0.01f, 0.9f, 0.999f, 1e-8f, 1e-2f, true, true),
                p -> new SGD(p, 0.1f, 0.9f, 1e-4f, true),
                p -> new AdaGrad(p, 0.05f, 1e-3f, 0, 0.1f, 1e-10f),
                p -> new AdaMax(p, 0.01f, 0.9f, 0.999f, 1e-8f, 0),
                p -> new AdaDelta(p, 1.0f, 0.9f, 1e-6f, 0),
                p -> new Lamb(p, 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f),
                p -> new Lars(p, 0.1f, 0.9f, 1e-4f, 0.001f, 1e-8f),
                p -> new NAdam(p, 0.01f, 0.9f, 0.999f, 1e-8f, 0, 0.004f),
                p -> new RMSProp(p, 0.01f, 0.99f, 1e-8f, 0, 0.9f, true));
    }

    private static Sequential mlp(long seed) {
        return new Sequential(new Linear(DIMS, 32, seed), new ReLU(), new Linear(32, 32, seed + 1), new ReLU(), new Linear(32, CLASSES, seed + 2));
    }

    private static float step(Sequential model, Optimizer opt, int b) {
        float[] xb = new float[BATCH * DIMS];
        int[] yb = new int[BATCH];
        DATA.batch(b % (DATA.count / BATCH), BATCH, xb, yb);
        Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, BATCH, DIMS)), Tensor.of(yb, BATCH));
        loss.backward();
        opt.step();
        float value = loss.item();
        opt.zeroGrad();
        return value;
    }

    /**
     * Training {@code 2k} steps in one go and training {@code k}, saving model, optimiser and
     * scheduler, loading them into a model and optimiser built from other seeds and training
     * {@code k} more give the same losses and weights, bit for bit.
     */
    static void resumeMatches(Function<List<Tensor>, Optimizer> make, Path dir, int k) {
        Sequential straight = mlp(1);
        Optimizer straightOpt = make.apply(straight.parameters());
        LRScheduler straightLr = LRScheduler.step(straightOpt, 3, 0.5f);
        List<Float> expected = new ArrayList<>();
        for (int b = 0; b < 2 * k; b++) {
            expected.add(step(straight, straightOpt, b));
            straightLr.step();
        }

        Sequential first = mlp(1);
        Optimizer firstOpt = make.apply(first.parameters());
        LRScheduler firstLr = LRScheduler.step(firstOpt, 3, 0.5f);
        for (int b = 0; b < k; b++) {
            step(first, firstOpt, b);
            firstLr.step();
        }
        Path file = dir.resolve("run.safetensors");
        new StateDict().put("model", first.stateDict()).put("optimizer", firstOpt.stateDict()).put("scheduler", firstLr.stateDict()).save(file);

        Sequential resumed = mlp(100);
        Optimizer resumedOpt = make.apply(resumed.parameters());
        LRScheduler resumedLr = LRScheduler.step(resumedOpt, 3, 0.5f);
        StateDict in = StateDict.load(file);
        resumed.loadStateDict(in.sub("model"));
        resumedOpt.loadStateDict(in.sub("optimizer"));
        resumedLr.loadStateDict(in.sub("scheduler"));
        String name = straightOpt.getClass().getSimpleName();
        for (int b = k; b < 2 * k; b++) {
            assertEquals(expected.get(b), step(resumed, resumedOpt, b), name + " step " + b);
            resumedLr.step();
        }
        for (int i = 0; i < straight.parameters().size(); i++) {
            assertArrayEquals(straight.parameters().get(i).toFloatArray(), resumed.parameters().get(i).toFloatArray(), name + " parameter " + i);
        }
    }

    /** Batch-norm running statistics are buffers: they are saved and restored with the model. */
    static void buffersRoundTrip(Path dir) {
        BatchNorm2d bn = new BatchNorm2d(4);
        bn.forward(Tensor.normal(5, 1.0f, 2.0f, 8, 4, 3, 3));
        Path file = dir.resolve("bn.safetensors");
        bn.stateDict().save(file);
        BatchNorm2d other = new BatchNorm2d(4);
        other.loadStateDict(StateDict.load(file));
        for (int i = 0; i < bn.buffers().size(); i++) {
            assertArrayEquals(bn.buffers().get(i).toFloatArray(), other.buffers().get(i).toFloatArray(), "buffer " + i);
        }
    }

    /** Loading into a model of another structure, or another optimiser type, fails before writing anything. */
    static void mismatchesAreRejected(Path dir) {
        Sequential model = mlp(1);
        Path file = dir.resolve("m.safetensors");
        new StateDict().put("model", model.stateDict()).put("optimizer", new Adam(model.parameters(), 0.01f).stateDict()).save(file);
        StateDict in = StateDict.load(file);
        Sequential wider = new Sequential(new Linear(DIMS, 64, 1), new ReLU(), new Linear(64, 32, 2), new ReLU(), new Linear(32, CLASSES, 3));
        assertThrows(IllegalArgumentException.class, () -> wider.loadStateDict(in.sub("model")));
        Sequential shorter = new Sequential(new Linear(DIMS, 32, 1), new ReLU(), new Linear(32, CLASSES, 3));
        assertThrows(IllegalArgumentException.class, () -> shorter.loadStateDict(in.sub("model")));
        assertThrows(IllegalArgumentException.class, () -> new SGD(model.parameters(), 0.1f).loadStateDict(in.sub("optimizer")));
    }
}
