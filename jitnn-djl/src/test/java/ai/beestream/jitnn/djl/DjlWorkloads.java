package ai.beestream.jitnn.djl;

import java.util.ArrayList;
import java.util.List;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

/** DJL programs written as DJL users write them; the tests run them on jitNN's host reference and devices. */
final class DjlWorkloads {

    private DjlWorkloads() {
    }

    /** Three Gaussian blobs in 4-D, labels 0..2: deterministic, linearly separable. */
    static float[][] blobs(int n) {
        float[] x = new float[n * 4];
        float[] y = new float[n];
        java.util.Random r = new java.util.Random(7);
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            y[i] = c;
            for (int j = 0; j < 4; j++) {
                x[i * 4 + j] = (float) (r.nextGaussian() * 0.5 + (j == c ? 3 : 0));
            }
        }
        return new float[][] { x, y };
    }

    /** Trains a two-layer MLP with DJL's Trainer and Adam; returns the loss of every step. */
    static List<Float> trainMlp(int steps, Optimizer optimizer) {
        SequentialBlock net = new SequentialBlock().add(Linear.builder().setUnits(32).build()).add(Activation::relu).add(Linear.builder().setUnits(3).build());
        return train(net, new Shape(48, 4), steps, optimizer, m -> {
            float[][] d = blobs(48);
            return new NDArray[] { m.create(d[0], new Shape(48, 4)), m.create(d[1], new Shape(48)) };
        });
    }

    /** Conv + BatchNorm + ReLU + max pooling + linear on 1x8x8 images, SGD with momentum. */
    static List<Float> trainCnn(int steps) {
        SequentialBlock net = new SequentialBlock().add(Conv2d.builder().setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).setFilters(6).build())
                .add(BatchNorm.builder().build()).add(Activation::relu).add(Pool.maxPool2dBlock(new Shape(2, 2))).add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(2).build());
        return train(net, new Shape(16, 1, 8, 8), steps, Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.05f)).optMomentum(0.9f).build(), m -> {
            float[] x = new float[16 * 64];
            float[] y = new float[16];
            for (int i = 0; i < 16; i++) {
                y[i] = i % 2;
                for (int p = 0; p < 64; p++) {
                    // class 1: bright left half
                    x[i * 64 + p] = (float) (Math.sin(i * 7 + p) * 0.3 + (y[i] == 1 && p % 8 < 4 ? 1 : 0));
                }
            }
            return new NDArray[] { m.create(x, new Shape(16, 1, 8, 8)), m.create(y, new Shape(16)) };
        });
    }

    /** An LSTM over [batch, time, features] then a linear head on the last step, Adam. */
    static List<Float> trainLstm(int steps) {
        SequentialBlock net = new SequentialBlock().add(LSTM.builder().setStateSize(8).setNumLayers(2).optBatchFirst(true).build())
                .add(new ai.djl.nn.LambdaBlock(l -> new NDList(l.head().get(":, -1, :")))).add(Linear.builder().setUnits(1).build());
        return train(net, new Shape(4, 5, 3), steps, Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.01f)).build(), m -> {
            float[] x = new float[4 * 5 * 3];
            float[] y = new float[4];
            for (int i = 0; i < x.length; i++) {
                x[i] = (float) Math.cos(i * 0.37);
            }
            for (int b = 0; b < 4; b++) {
                y[b] = b * 0.25f;
            }
            return new NDArray[] { m.create(x, new Shape(4, 5, 3)), m.create(y, new Shape(4, 1)) };
        }, Loss.l2Loss());
    }

    interface Data {
        NDArray[] make(NDManager manager);
    }

    static List<Float> train(SequentialBlock net, Shape input, int steps, Optimizer optimizer, Data data) {
        return train(net, input, steps, optimizer, data, Loss.softmaxCrossEntropyLoss());
    }

    static List<Float> train(SequentialBlock net, Shape input, int steps, Optimizer optimizer, Data data, Loss loss) {
        List<Float> losses = new ArrayList<>();
        // the same initial weights on every backend
        ai.djl.engine.Engine.getInstance().setRandomSeed(1234);
        try (Model model = Model.newInstance("net")) {
            model.setBlock(net);
            DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(optimizer).optInitializer(new XavierInitializer(), ai.djl.nn.Parameter.Type.WEIGHT);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(input);
                NDManager m = trainer.getManager();
                NDArray[] xy = data.make(m);
                for (int s = 0; s < steps; s++) {
                    try (GradientCollector gc = trainer.newGradientCollector()) {
                        NDList pred = trainer.forward(new NDList(xy[0]));
                        NDArray l = trainer.getLoss().evaluate(new NDList(xy[1]), pred);
                        gc.backward(l);
                        losses.add(l.getFloat());
                    }
                    trainer.step();
                }
            }
        }
        return losses;
    }
}
