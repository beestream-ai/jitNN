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

import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.classification.ClassificationException;
import javax.visrec.ml.model.ModelProvider;

import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Module;

/** A JSR 381 binary classifier on a jitNN network: the probability of the positive class for a feature vector. */
public final class JitnnBinaryClassifier implements BinaryClassifier<float[]>, ModelProvider<Module> {

    private final Module model;
    private final int inputs;
    private float threshold;

    JitnnBinaryClassifier(Module model, int inputs, float threshold) {
        this.model = model;
        this.inputs = inputs;
        this.threshold = threshold;
    }

    @Override
    public Float classify(float[] input) throws ClassificationException {
        if (input.length != inputs) {
            throw new ClassificationException(input.length + " features for a network of " + inputs + " inputs");
        }
        try {
            return Grad.inference(() -> model.forward(Tensor.of(input.clone(), 1, inputs)).sigmoid()).toFloatArray()[0];
        } catch (RuntimeException e) {
            throw new ClassificationException("classification failed: " + e.getMessage(), e);
        }
    }

    /** True when the probability reaches the builder's threshold. */
    public boolean isPositive(float[] input) {
        return classify(input) >= threshold;
    }

    @Override
    public Module getModel() {
        return model;
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }
}
