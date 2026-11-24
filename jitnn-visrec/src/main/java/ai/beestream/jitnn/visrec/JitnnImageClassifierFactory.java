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

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.ml.model.ModelCreationException;
import javax.visrec.spi.ImageClassifierFactory;

import ai.beestream.jitnn.io.StateDict;

/**
 * Builds {@code NeuralNetImageClassifier}s for {@link BufferedImage} inputs, as the builder's
 * settings ask, the way JSR 381's reference implementation reads them:
 * <ul>
 * <li>{@code networkArchitecture}: a network in Deep Netts' JSON format ({@link DeepNettsArchitecture});</li>
 * <li>{@code labelsFile}: one class label per line;</li>
 * <li>{@code trainingFile}: an index of {@code <image path> <label>} lines, paths relative to the
 * index file's folder (a line with a path only takes its folder's name as the label);</li>
 * <li>{@code learningRate}, {@code maxEpochs}, {@code maxError}, {@code threshold};</li>
 * <li>{@code exportModel}: where the trained model is saved, one safetensors file holding the
 * weights, the architecture and the labels; {@code importModel} reads such a file instead of
 * training.</li>
 * </ul>
 */
public final class JitnnImageClassifierFactory implements ImageClassifierFactory<BufferedImage> {

    private static final Logger LOG = Logger.getLogger(JitnnImageClassifierFactory.class.getName());

    static final String ARCHITECTURE = "visrec.architecture";
    static final String LABELS = "visrec.labels";

    @Override
    public Class<BufferedImage> getImageClass() {
        return BufferedImage.class;
    }

    @Override
    public ImageClassifier<BufferedImage> create(NeuralNetImageClassifier.BuildingBlock<BufferedImage> block) throws ModelCreationException {
        try {
            return block.getImportPath() != null ? load(block) : train(block);
        } catch (IOException | RuntimeException e) {
            throw new ModelCreationException("jitNN could not build the image classifier: " + e.getMessage(), e);
        }
    }

    private static ImageClassifier<BufferedImage> load(NeuralNetImageClassifier.BuildingBlock<BufferedImage> block) throws IOException {
        StateDict saved = StateDict.load(block.getImportPath());
        DeepNettsArchitecture network = DeepNettsArchitecture.parse(saved.value(ARCHITECTURE));
        String[] labels = saved.value(LABELS).split("\n");
        StateDict weights = new StateDict();
        for (String name : saved.tensorNames()) {
            weights.put(name, saved.tensor(name));
        }
        network.model.loadStateDict(weights);
        network.model.train(false);
        return new JitnnImageClassifier(network, labels, block.getThreshold());
    }

    private static ImageClassifier<BufferedImage> train(NeuralNetImageClassifier.BuildingBlock<BufferedImage> block) throws IOException {
        if (block.getNetworkArchitecture() == null || block.getLabelsPath() == null || block.getTrainingPath() == null) {
            throw new IOException("training needs networkArchitecture, labelsFile and trainingFile (or importModel)");
        }
        DeepNettsArchitecture network = DeepNettsArchitecture.read(block.getNetworkArchitecture());
        String[] labels = Files.readAllLines(block.getLabelsPath()).stream().map(String::trim).filter(s -> !s.isEmpty()).toArray(String[]::new);
        if (labels.length != network.outputs) {
            throw new IOException(labels.length + " labels for an output layer of " + network.outputs);
        }
        // the builder's image size wins over the architecture's input layer, as in the reference implementation
        int width = block.getImageWidth() > 0 ? block.getImageWidth() : network.width;
        int height = block.getImageHeight() > 0 ? block.getImageHeight() : network.height;
        if (width != network.width || height != network.height) {
            throw new IOException("image size " + width + "x" + height + " differs from the network's input layer " + network.width + "x" + network.height);
        }
        Path index = block.getTrainingPath();
        Path root = index.toAbsolutePath().getParent();
        List<String> lines = Files.readAllLines(index);
        List<float[]> images = new ArrayList<>();
        List<Integer> classes = new ArrayList<>();
        List<String> labelList = Arrays.asList(labels);
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] parts = line.split("\\s+");
            Path image = root.resolve(parts[0]);
            String label = parts.length > 1 ? parts[1] : image.getParent().getFileName().toString();
            int c = labelList.indexOf(label);
            if (c < 0) {
                throw new IOException("label " + label + " of " + parts[0] + " is not in the labels file");
            }
            images.add(Images.pixels(JitnnImageFactoryService.BUFFERED_IMAGES.getImage(image), network.width, network.height, network.channels));
            classes.add(c);
        }
        int n = images.size();
        int size = network.channels * network.height * network.width;
        float[] x = new float[n * size];
        for (int i = 0; i < n; i++) {
            System.arraycopy(images.get(i), 0, x, i * size, size);
        }
        LOG.info("training on " + n + " images, " + labels.length + " classes");
        int[] shape = { network.channels, network.height, network.width };
        if (network.softmax) {
            Training.train(network.model, x, n, shape, classes.stream().mapToInt(Integer::intValue).toArray(), null, network.outputs, block.getLearningRate(), 0.7f,
                    block.getMaxEpochs(), block.getMaxError());
        } else {
            float[] targets = new float[n * network.outputs];
            for (int i = 0; i < n; i++) {
                targets[i * network.outputs + classes.get(i)] = 1f;
            }
            Training.train(network.model, x, n, shape, null, targets, network.outputs, block.getLearningRate(), 0.7f, block.getMaxEpochs(), block.getMaxError());
        }
        if (block.getExportPath() != null) {
            StateDict d = network.model.stateDict();
            d.put(ARCHITECTURE, network.json);
            d.put(LABELS, String.join("\n", labels));
            d.save(block.getExportPath());
        }
        return new JitnnImageClassifier(network, labels, block.getThreshold());
    }
}
