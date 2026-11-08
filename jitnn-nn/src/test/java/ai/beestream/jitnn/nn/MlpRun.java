package ai.beestream.jitnn.nn;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.Blobs;
import ai.beestream.jitnn.nn.optim.Adam;
import ai.beestream.jitnn.nn.optim.Optimizer;
import ai.beestream.jitnn.nn.optim.SGD;

/** One MLP training run on the synthetic blobs, shared by the reference and device tests. */
final class MlpRun {

    static final int DIMS = 16;
    static final int CLASSES = 3;
    static final int BATCH = 64;

    final List<Float> losses = new ArrayList<>();
    float accuracy;
    float[] firstWeights;

    static MlpRun train(boolean adam, int epochs) {
        MlpRun run = new MlpRun();
        Blobs train = new Blobs(512, DIMS, CLASSES, 0.8f, 11);
        Blobs test = new Blobs(256, DIMS, CLASSES, 0.8f, 11);
        Sequential model = new Sequential(new Linear(DIMS, 32, 1), new ReLU(), new Linear(32, 32, 2), new ReLU(), new Linear(32, CLASSES, 3));
        Optimizer opt = adam ? new Adam(model.parameters(), 0.01f) : new SGD(model.parameters(), 0.1f, 0.9f, 1e-4f, false);
        float[] xb = new float[BATCH * DIMS];
        int[] yb = new int[BATCH];
        for (int e = 0; e < epochs; e++) {
            for (int b = 0; b < train.count / BATCH; b++) {
                train.batch(b, BATCH, xb, yb);
                Tensor loss = F.crossEntropy(model.forward(Tensor.of(xb, BATCH, DIMS)), Tensor.of(yb, BATCH));
                loss.backward();
                opt.step();
                run.losses.add(loss.item());
                opt.zeroGrad();
            }
        }
        run.accuracy = F.accuracy(model.forward(Tensor.of(test.x, test.count, DIMS)), Tensor.of(test.y, test.count));
        run.firstWeights = model.parameters().get(0).toFloatArray();
        return run;
    }
}
