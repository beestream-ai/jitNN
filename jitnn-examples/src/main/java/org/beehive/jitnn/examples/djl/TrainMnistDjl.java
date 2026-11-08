package org.beehive.jitnn.examples.djl;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;

/**
 * DJL code on jitNN: the model, dataset, trainer and training loop of DJL's TrainMnist example,
 * written against DJL's API only. With jitnn-djl on the class path, jitNN is DJL's default engine,
 * so nothing here names it. DJL downloads MNIST into its own cache on first use.
 *
 * <pre>run.sh djl.TrainMnistDjl [epochs]</pre>
 */
public final class TrainMnistDjl {

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 2;
        int batch = 64;
        System.out.println("DJL engine: " + Engine.getInstance().getEngineName());

        Mnist train = Mnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(batch, true).optLimit(Long.MAX_VALUE).build();
        Mnist test = Mnist.builder().optUsage(Dataset.Usage.TEST).setSampling(batch, true).build();
        train.prepare();
        test.prepare();

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .optOptimizer(Optimizer.adam().build())
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 28 * 28));
                long t0 = System.nanoTime();
                EasyTrain.fit(trainer, epochs, train, test);
                TrainingResult result = trainer.getTrainingResult();
                System.out.printf("%d epochs in %.1f s: train loss %.4f, validation accuracy %.2f%%%n", epochs, (System.nanoTime() - t0) / 1e9,
                        result.getTrainLoss(), 100 * result.getValidateEvaluation("Accuracy"));
            }
        }
        System.exit(0);
    }
}
