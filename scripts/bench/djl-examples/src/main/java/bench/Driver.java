package bench;

import java.util.Arrays;

import ai.djl.engine.Engine;
import ai.djl.training.TrainingResult;

/** Runs one of DJL's training examples, unchanged, and reports wall time, loss and accuracy. */
public final class Driver {

    public static void main(String[] args) throws Exception {
        String example = args[0];
        String[] rest = Arrays.copyOfRange(args, 1, args.length);
        System.out.println("[djlex] engine " + Engine.getInstance() + ", GPUs " + Engine.getInstance().getGpuCount());
        long t0 = System.nanoTime();
        TrainingResult r = switch (example) {
            case "mnist" -> ai.djl.examples.training.TrainMnist.runExample(rest);
            case "lstm" -> ai.djl.examples.training.TrainMnistWithLSTM.runExample(rest);
            case "resnet" -> ai.djl.examples.training.transferlearning.TrainResnetWithCifar10.runExample(rest);
            default -> throw new IllegalArgumentException(example);
        };
        double s = (System.nanoTime() - t0) / 1e9;
        Float acc = r.getValidateEvaluation("Accuracy");
        System.out.printf("[djlex] %s engine=%s total %.1f s epochs %d train loss %.4f validation accuracy %s%n", example, Engine.getInstance().getEngineName(), s, r.getEpoch(),
                r.getTrainLoss(), acc == null ? "-" : String.format("%.4f", acc));
        System.exit(0);
    }
}
