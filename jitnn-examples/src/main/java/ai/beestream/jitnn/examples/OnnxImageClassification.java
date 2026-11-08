package ai.beestream.jitnn.examples;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.onnx.OnnxSession;

/**
 * Classifies a photo with ResNet-50 from the ONNX model zoo, after DJL's ImageClassification
 * example: the ONNX file runs on jitNN's kernels, no ONNX Runtime involved. Downloads the model
 * (98 MB), the ImageNet class names and, without an argument, DJL's kitten photo.
 *
 * <pre>run.sh OnnxImageClassification [image file]</pre>
 */
public final class OnnxImageClassification {

    public static void main(String[] args) throws Exception {
        Path dir = Examples.dataDir().resolve("onnx");
        Path model = Examples.download("https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx", dir.resolve("resnet50-v1-7.onnx"));
        List<String> classes = Files.readAllLines(Examples.download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", dir.resolve("imagenet_classes.txt")));
        Path image = args.length > 0 ? Path.of(args[0]) : Examples.download("https://resources.djl.ai/images/kitten.jpg", dir.resolve("kitten.jpg"));

        Tensor input = Tensor.of(preprocess(ImageIO.read(image.toFile())), 1, 3, 224, 224);
        try (OnnxSession session = OnnxSession.load(model)) {
            String in = session.inputNames().get(0);
            String out = session.outputNames().get(0);

            long t0 = System.nanoTime();
            float[] logits = session.run(Map.of(in, input)).get(out).toFloatArray();   // compiles the graph
            System.out.printf("first call (loads and compiles the model's plan): %.2f s%n", Examples.seconds(t0));

            float[] p = Examples.softmax(logits, 0, logits.length, 1);
            System.out.println("top 5 for " + image.getFileName() + ":");
            for (int c : Examples.topK(p, 5)) {
                System.out.printf("  %5.1f%%  %s%n", 100 * p[c], classes.get(c));
            }

            double[] ms = new double[20];
            for (int i = 0; i < ms.length; i++) {
                long t = System.nanoTime();
                session.run(Map.of(in, input)).get(out).toFloatArray();    // replays the compiled plan
                ms[i] = (System.nanoTime() - t) / 1e6;
            }
            Arrays.sort(ms);
            System.out.printf("later calls: median %.2f ms over %d calls%n", ms[ms.length / 2], ms.length);
        }
    }

    /** Shorter side to 256, centre crop 224, RGB normalised with ImageNet's mean and std, CHW. */
    static float[] preprocess(BufferedImage src) {
        double scale = 256.0 / Math.min(src.getWidth(), src.getHeight());
        int w = (int) Math.round(src.getWidth() * scale);
        int h = (int) Math.round(src.getHeight() * scale);
        BufferedImage resized = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, w, h, null);
        g.dispose();
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };
        int x0 = (w - 224) / 2;
        int y0 = (h - 224) / 2;
        float[] chw = new float[3 * 224 * 224];
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int rgb = resized.getRGB(x0 + x, y0 + y);
                for (int c = 0; c < 3; c++) {
                    float v = ((rgb >> (16 - 8 * c)) & 0xFF) / 255f;
                    chw[c * 224 * 224 + y * 224 + x] = (v - mean[c]) / std[c];
                }
            }
        }
        return chw;
    }
}
