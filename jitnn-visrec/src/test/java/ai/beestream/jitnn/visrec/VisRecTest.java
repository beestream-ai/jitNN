/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.visrec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetBinaryClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.spi.ServiceProvider;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** jitNN through the JSR 381 API only: the provider lookup and the two builders. */
class VisRecTest {

    @TempDir
    Path dir;

    @Test
    void jitnnIsTheServiceProvider() {
        assertEquals("jitNN", ServiceProvider.current().getImplementationService().getName());
        assertTrue(ServiceProvider.current().getImageFactoryService().getByImageType(BufferedImage.class).isPresent());
    }

    @Test
    void binaryClassifierFromCsv() throws Exception {
        Random r = new Random(3);
        StringBuilder csv = new StringBuilder("x,y,above\n");
        for (int i = 0; i < 400; i++) {
            float x = r.nextFloat();
            float y = r.nextFloat();
            csv.append(x).append(',').append(y).append(',').append(x + y > 1 ? 1 : 0).append('\n');
        }
        Path file = dir.resolve("points.csv");
        Files.writeString(file, csv);

        BinaryClassifier<float[]> classifier = NeuralNetBinaryClassifier.builder()
                .inputClass(float[].class)
                .inputsNum(2)
                .hiddenLayers(8)
                .maxEpochs(400)
                .learningRate(2f)
                .threshold(0.5f)
                .trainingPath(file)
                .build();

        int correct = 0;
        for (int i = 0; i < 200; i++) {
            float x = r.nextFloat();
            float y = r.nextFloat();
            boolean above = classifier.classify(new float[] { x, y }) >= 0.5f;
            correct += above == (x + y > 1) ? 1 : 0;
        }
        System.out.printf("[visrec] binary classifier: %d/200 correct%n", correct);
        assertTrue(correct >= 180, correct + "/200");
    }

    /** Red, green and blue squares on noise; a small CNN in Deep Netts' format, trained, exported and imported. */
    @Test
    void imageClassifierTrainsExportsAndImports() throws Exception {
        String[] labels = { "red", "green", "blue" };
        Color[] colors = { Color.RED, Color.GREEN, Color.BLUE };
        Random r = new Random(7);
        List<String> index = new ArrayList<>();
        for (int c = 0; c < 3; c++) {
            Files.createDirectories(dir.resolve(labels[c]));
            for (int i = 0; i < 40; i++) {
                String name = labels[c] + "/" + i + ".png";
                ImageIO.write(square(colors[c], r), "png", dir.resolve(name).toFile());
                index.add(name + " " + labels[c]);
            }
        }
        Path indexFile = dir.resolve("index.txt");
        Files.write(indexFile, index);
        Path labelsFile = dir.resolve("labels.txt");
        Files.write(labelsFile, List.of(labels));
        Path arch = dir.resolve("network.json");
        Files.writeString(arch, """
                {"networkType": "CONVOLUTIONAL", "lossFunction": "CROSS_ENTROPY", "layers": [
                  {"layerType": "INPUT", "width": 16, "height": 16, "channels": 3},
                  {"layerType": "CONVOLUTIONAL", "filterWidth": 3, "filterHeight": 3, "channels": 4, "stride": 1, "activation": "RELU"},
                  {"layerType": "MAXPOOLING", "filterWidth": 2, "filterHeight": 2, "stride": 2},
                  {"layerType": "FULLY_CONNECTED", "width": 16, "activation": "RELU"},
                  {"layerType": "OUTPUT", "width": 3, "activation": "SOFTMAX"}]}
                """);
        Path model = dir.resolve("model.safetensors");

        ImageClassifier<BufferedImage> trained = NeuralNetImageClassifier.builder()
                .inputClass(BufferedImage.class)
                .imageWidth(16)
                .imageHeight(16)
                .labelsFile(labelsFile)
                .trainingFile(indexFile)
                .networkArchitecture(arch)
                .learningRate(0.05f)
                .maxEpochs(40)
                .maxError(0.01f)
                .threshold(0f)
                .exportModel(model)
                .build();
        assertTrue(Files.size(model) > 0);

        ImageClassifier<BufferedImage> imported = NeuralNetImageClassifier.builder()
                .inputClass(BufferedImage.class)
                .importModel(model)
                .threshold(0f)
                .build();

        for (int c = 0; c < 3; c++) {
            Path test = dir.resolve("test-" + labels[c] + ".png");
            ImageIO.write(square(colors[c], r), "png", test.toFile());
            Map<String, Float> a = trained.classify(test);
            Map<String, Float> b = imported.classify(test);
            String best = a.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
            System.out.printf("[visrec] %s image: %s%n", labels[c], a);
            assertEquals(labels[c], best);
            for (String l : labels) {
                assertEquals(a.get(l), b.get(l), 1e-5f, "imported model, " + l);
            }
        }
    }

    private static BufferedImage square(Color color, Random r) {
        BufferedImage img = new BufferedImage(20, 20, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 20; y++) {
            for (int x = 0; x < 20; x++) {
                int g = r.nextInt(60);
                img.setRGB(x, y, new Color(g, g, g).getRGB());
            }
        }
        var g2 = img.createGraphics();
        g2.setColor(color);
        int x0 = 2 + r.nextInt(6);
        int y0 = 2 + r.nextInt(6);
        g2.fillRect(x0, y0, 10, 10);
        g2.dispose();
        return img;
    }
}
