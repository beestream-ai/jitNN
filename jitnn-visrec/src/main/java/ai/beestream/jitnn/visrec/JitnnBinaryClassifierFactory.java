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

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.classification.NeuralNetBinaryClassifier;
import javax.visrec.ml.model.ModelCreationException;
import javax.visrec.spi.BinaryClassifierFactory;

import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.Sigmoid;

/**
 * Builds {@code NeuralNetBinaryClassifier}s over {@code float[]} feature vectors as the reference
 * implementation does: {@code inputsNum} inputs, fully connected hidden layers of the given widths
 * with sigmoid activations, one sigmoid output and cross-entropy loss, trained with SGD on a CSV
 * file (a header line, then {@code inputsNum} feature columns and the 0/1 target).
 */
public final class JitnnBinaryClassifierFactory implements BinaryClassifierFactory<float[]> {

    @Override
    public Class<float[]> getTargetClass() {
        return float[].class;
    }

    @Override
    public BinaryClassifier<float[]> create(NeuralNetBinaryClassifier.BuildingBlock<float[]> block) throws ModelCreationException {
        try {
            int inputs = block.getInputsNum();
            if (inputs <= 0 || block.getTrainingPath() == null) {
                throw new IOException("needs inputsNum and trainingPath");
            }
            List<Module> layers = new ArrayList<>();
            int width = inputs;
            long seed = 1;
            int[] hidden = block.getHiddenLayers() == null ? new int[0] : block.getHiddenLayers();
            for (int h : hidden) {
                layers.add(new Linear(width, h, seed++));
                layers.add(new Sigmoid());
                width = h;
            }
            layers.add(new Linear(width, 1, seed));
            Module model = new Sequential(layers.toArray(new Module[0]));

            List<String> lines = Files.readAllLines(block.getTrainingPath());
            List<float[]> rows = new ArrayList<>();
            for (int i = 1; i < lines.size(); i++) {        // the first line holds the column names
                String line = lines.get(i).trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] cells = line.split(",");
                if (cells.length < inputs + 1) {
                    throw new IOException("line " + (i + 1) + " has " + cells.length + " columns, needs " + (inputs + 1));
                }
                float[] r = new float[inputs + 1];
                for (int k = 0; k <= inputs; k++) {
                    r[k] = Float.parseFloat(cells[k].trim());
                }
                rows.add(r);
            }
            int n = rows.size();
            float[] x = new float[n * inputs];
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(rows.get(i), 0, x, i * inputs, inputs);
                y[i] = rows.get(i)[inputs];
            }
            Training.train(model, x, n, new int[] { inputs }, null, y, 1, block.getLearningRate(), 0f, block.getMaxEpochs(), block.getMaxError());
            return new JitnnBinaryClassifier(model, inputs, block.getThreshold());
        } catch (IOException | RuntimeException e) {
            throw new ModelCreationException("jitNN could not build the binary classifier: " + e.getMessage(), e);
        }
    }
}
