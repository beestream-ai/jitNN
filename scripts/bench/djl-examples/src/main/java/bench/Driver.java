package bench;

import java.util.Arrays;

import ai.djl.engine.Engine;
import ai.djl.training.TrainingResult;

/**
 * Runs one of DJL's examples. Training examples report wall time, loss and accuracy; inference
 * examples report their result and the time of each call ({@code -Drepeat=N} calls, each loading
 * the model as the example does). {@code yolo-loop} times one predictor over many images.
 */
public final class Driver {

    public static void main(String[] args) throws Exception {
        String example = args[0];
        String[] rest = Arrays.copyOfRange(args, 1, args.length);
        System.out.println("[djlex] engine " + Engine.getInstance() + ", GPUs " + Engine.getInstance().getGpuCount());
        switch (example) {
            case "classify", "yolo" -> {
                infer(example);
                System.exit(0);
            }
            case "classify-debug" -> ClassifyDebug.main(rest);
            case "yolo-loop" -> {
                yoloLoop(Integer.getInteger("repeat", 50));
                System.exit(0);
            }
            default -> {
            }
        }
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
        profile();
        System.exit(0);
    }

    private static void infer(String example) throws Exception {
        int repeat = Integer.getInteger("repeat", 5);
        for (int i = 0; i < repeat; i++) {
            long t0 = System.nanoTime();
            Object result = example.equals("classify") ? ai.djl.examples.inference.cv.ImageClassification.predict()
                    : ai.djl.examples.inference.cv.YoloDetection.predict();
            double ms = (System.nanoTime() - t0) / 1e6;
            if (i == 0) {
                System.out.println("[djlex] " + example + " result: " + String.valueOf(result).replaceAll("\\s+", " "));
            }
            System.out.printf("[djlex] %s engine=%s call %d %.1f ms%n", example, Engine.getInstance().getEngineName(), i, ms);
        }
        profile();
    }

    /** YoloDetection's criteria, loaded once, then {@code n} predictions on its test image. */
    private static void yoloLoop(int n) throws Exception {
        java.nio.file.Path img = java.nio.file.Paths.get("src/test/resources/yolov8_test.jpg");
        ai.djl.repository.zoo.Criteria<java.nio.file.Path, ai.djl.modality.cv.output.DetectedObjects> criteria = ai.djl.repository.zoo.Criteria.builder()
                .setTypes(java.nio.file.Path.class, ai.djl.modality.cv.output.DetectedObjects.class).optModelUrls("djl://ai.djl.onnxruntime/yolo11n")
                .optEngine(Engine.getDefaultEngineName()).optArgument("width", 640).optArgument("height", 640).optArgument("resize", true)
                .optArgument("toTensor", true).optArgument("applyRatio", true).optArgument("threshold", 0.6f).optArgument("maxBox", 1000)
                .optTranslatorFactory(new ai.djl.modality.cv.translator.YoloV8TranslatorFactory()).build();
        try (var model = criteria.loadModel(); var predictor = model.newPredictor()) {
            double[] ms = new double[n];
            Object first = null;
            for (int i = 0; i < n; i++) {
                long t0 = System.nanoTime();
                Object r = predictor.predict(img);
                ms[i] = (System.nanoTime() - t0) / 1e6;
                first = first == null ? r : first;
            }
            System.out.println("[djlex] yolo-loop result: " + String.valueOf(first).replaceAll("\\s+", " "));
            double[] steady = Arrays.copyOfRange(ms, Math.min(5, n - 1), n);
            Arrays.sort(steady);
            System.out.printf("[djlex] yolo-loop engine=%s first %.1f ms, median of last %d %.2f ms, min %.2f ms%n", Engine.getInstance().getEngineName(), ms[0],
                    steady.length, steady[steady.length / 2], steady[0]);
        }
        profile();
    }

    private static void profile() throws Exception {
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
    }
}
