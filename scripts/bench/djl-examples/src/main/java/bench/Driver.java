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
        if (Boolean.getBoolean("jitnn.profile")) {
            // jitNN's device time per operator task over the whole run (jitNN engine only)
            Class<?> engine = Class.forName("org.beehive.jitnn.engine.Engine");
            @SuppressWarnings("unchecked")
            java.util.Map<String, long[]> k = (java.util.Map<String, long[]>) engine.getMethod("kernelBreakdown").invoke(null);
            long total = k.values().stream().mapToLong(v -> v[1]).sum();
            System.out.printf("[djlex] device time %.1f ms in %d kernels%n", total / 1e6, k.size());
            k.entrySet().stream().sorted((p, q) -> Long.compare(q.getValue()[1], p.getValue()[1])).limit(40)
                    .forEach(e -> System.out.printf("[djlex]   %-44s %8d launches %10.1f ms%n", e.getKey(), e.getValue()[0], e.getValue()[1] / 1e6));
            long[] split = (long[]) engine.getMethod("timeSplit").invoke(null);
            System.out.printf("[djlex] host: jitNN executor %.1f ms, TornadoVM execute() %.1f ms%n", split[0] / 1e6, split[1] / 1e6);
            long[] tr = (long[]) engine.getMethod("transferStats").invoke(null);
            System.out.printf("[djlex] traffic: %d fetches (%.1f MB), %d persistent re-uploads (%.1f MB), %d input copies (%.1f MB)%n", tr[0], tr[1] * 4 / 1e6, tr[2],
                    tr[3] * 4 / 1e6, tr[4], tr[5] * 4 / 1e6);
            long[] st = (long[]) engine.getMethod("planStats").invoke(null);
            System.out.printf("[djlex] plans: %d compiled, %d replays%n", st[0], st[1]);
        }
        System.exit(0);
    }
}
