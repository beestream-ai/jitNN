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
import java.io.InputStream;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.visrec.ml.classification.ClassificationException;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.model.ModelProvider;

import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Module;

/**
 * A JSR 381 image classifier on a jitNN model: the image is scaled to the network's input, run
 * through it, and every label whose score reaches the threshold is returned with its score
 * (softmax or sigmoid scores, as the architecture's output layer says).
 */
public final class JitnnImageClassifier implements ImageClassifier<BufferedImage>, ModelProvider<Module> {

    private final DeepNettsArchitecture network;
    private final String[] labels;
    private float threshold;

    JitnnImageClassifier(DeepNettsArchitecture network, String[] labels, float threshold) {
        this.network = network;
        this.labels = labels.clone();
        this.threshold = threshold;
    }

    @Override
    public Map<String, Float> classify(BufferedImage input) throws ClassificationException {
        try {
            float[] x = Images.pixels(input, network.width, network.height, network.channels);
            Tensor in = Tensor.of(x, 1, network.channels, network.height, network.width);
            float[] scores = Grad.inference(() -> {
                Tensor logits = network.model.forward(in);
                return network.softmax ? logits.softmax() : logits.sigmoid();
            }).toFloatArray();
            Map<String, Float> result = new LinkedHashMap<>();
            for (int i = 0; i < scores.length; i++) {
                if (scores[i] >= threshold) {
                    result.put(labels[i], scores[i]);
                }
            }
            return result;
        } catch (RuntimeException e) {
            throw new ClassificationException("classification failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Map<String, Float> classify(Path input) throws ClassificationException {
        try {
            return classify(JitnnImageFactoryService.BUFFERED_IMAGES.getImage(input));
        } catch (IOException e) {
            throw new ClassificationException("could not read " + input, e);
        }
    }

    @Override
    public Map<String, Float> classify(InputStream input) throws ClassificationException {
        try {
            return classify(JitnnImageFactoryService.BUFFERED_IMAGES.getImage(input));
        } catch (IOException e) {
            throw new ClassificationException("could not read the image stream", e);
        }
    }

    /** The jitNN model behind the classifier (to train further, export or inspect). */
    @Override
    public Module getModel() {
        return network.model;
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    /** The class labels, in output order. */
    public String[] getLabels() {
        return labels.clone();
    }
}
